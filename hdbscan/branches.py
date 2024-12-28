# Support branch detection within clusters.
import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from scipy.sparse import coo_array
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from joblib import Memory
from joblib import Parallel, delayed
from joblib.parallel import cpu_count
from ._hdbscan_linkage import label
from .plots import CondensedTree, SingleLinkageTree, ApproximationGraph
from .prediction import approximate_predict
from ._hdbscan_tree import recurse_leaf_dfs
from .hdbscan_ import _tree_to_labels


def detect_branches_in_clusters(
    clusterer,
    min_branch_size=None,
    allow_single_branch=False,
    branch_detection_method="full",
    branch_selection_method="eom",
    branch_selection_persistence=0.0,
    max_branch_size=0,
    label_sides_as_branches=False,
):
    """
    Performs a flare-detection post-processing step to detect branches within
    clusters [1]_.

    For each cluster, a graph is constructed connecting the data points based on
    their mutual reachability distances. Each edge is given a centrality value
    based on how far it lies from the cluster's center. Then, the edges are
    clustered as if that centrality was a distance, progressively removing the
    'center' of each cluster and seeing how many branches remain.

    Parameters
    ----------

    clusterer : hdbscan.HDBSCAN
        The clusterer object that has been fit to the data with branch detection
        data generated.

    min_branch_size : int, optional (default=None)
        The minimum number of samples in a group for that group to be
        considered a branch; groupings smaller than this size will seen as
        points falling out of a branch. Defaults to the clusterer's min_cluster_size.

    allow_single_branch : bool, optional (default=False)
        Analogous to ``allow_single_cluster``.

    branch_detection_method : str, optional (default=``full``)
        Deteremines which graph is conctructed to detect branches with. Valid
        values are, ordered by increasing computation cost and decreasing
        sensitivity to noise:
        - ``core``: Contains the edges that connect each point to all other
          points within a mutual reachability distance lower than or equal to
          the point's core distance. This is the cluster's subgraph of the
          k-NN graph over the entire data set (with k = ``min_samples``).
        - ``full``: Contains all edges between points in each cluster with a
          mutual reachability distance lower than or equal to the distance of
          the most-distance point in each cluster. These graphs represent the
          0-dimensional simplicial complex of each cluster at the first point in
          the filtration where they contain all their points.

    branch_selection_method : str, optional (default='eom')
        The method used to select branches from the cluster's condensed tree.
        The standard approach for FLASC is to use the ``eom`` approach.
        Options are:
          * ``eom``
          * ``leaf``

    branch_selection_epsilon: float, optional (default=0.0)
        A lower epsilon threshold. Only branches with a death above this value
        will be considered. See [3]_ for more information. Note that this
        should not be used if we want to predict the cluster labels for new
        points in future (e.g. using approximate_predict), as the
        :func:`~hdbscan.branches.approximate_predict` function is not aware of
        this argument.

    branch_selection_persistence: float, optional (default=0.0)
        An eccentricity persistence threshold. Branches with a persistence below
        this value will be merged. See [3]_ for more information. Note that this
        should not be used if we want to predict the cluster labels for new
        points in future (e.g. using approximate_predict), as the
        :func:`~hdbscan.branches.approximate_predict` function is not aware of
        this argument.

    max_branch_size : int, optional (default=0)
        A limit to the size of clusters returned by the ``eom`` algorithm.
        Has no effect when using ``leaf`` clustering (where clusters are
        usually small regardless). Note that this should not be used if we
        want to predict the cluster labels for new points in future (e.g. using
        :func:`~hdbscan.branches.approximate_predict`), as that function is
        not aware of this argument.

    label_sides_as_branches : bool, optional (default=False),
        When this flag is False, branches are only labelled for clusters with at
        least three branches (i.e., at least y-shapes). Clusters with only two
        branches represent l-shapes. The two branches describe the cluster's
        outsides growing towards each other. Enabling this flag separates these
        branches from each other in the produced labelling.

    Returns
    -------
    labels : np.ndarray, shape (n_samples, )
        Labels that differentiate all subgroups (clusters and branches). Noisy
        samples are given the label -1.

    probabilities : np.ndarray, shape (n_samples, )
        Probabilities considering both cluster and branch membership. Noisy
        samples are assigned 0.

    branch_labels : np.ndarray, shape (n_samples, )
        Branch labels for each point. Noisy samples are given the label -1.

    branch_probabilities : np.ndarray, shape (n_samples, )
        Branch membership strengths for each point. Noisy samples are
        assigned 0.

    branch_persistences : tuple (n_clusters)
        A branch persistence (eccentricity range) for each detected branch.

    approximation_graphs : tuple (n_clusters)
        The graphs used to detect branches in each cluster stored as a numpy
        array with four columns: source, target, centrality, mutual reachability
        distance. Points are labelled by their row-index into the input data.
        The edges contained in the graphs depend on the ``branch_detection_method``:
        - ``core``: Contains the edges that connect each point to all other
          points in a cluster within a mutual reachability distance lower than
          or equal to the point's core distance. This is an extension of the
          minimum spanning tree introducing only edges with equal distances. The
          reachability distance introduces ``num_points`` * ``min_samples`` of
          such edges.
        - ``full``: Contains all edges between points in each cluster with a
          mutual reachability distance lower than or equal to the distance of
          the most-distance point in each cluster. These graphs represent the
          0-dimensional simplicial complex of each cluster at the first point in
          the filtration where they contain all their points.

    condensed_trees : tuple (n_clusters)
        A condensed branch hierarchy for each cluster produced during the
        branch detection step. Data points are numbered with in-cluster ids.

    linkage_trees : tuple (n_clusters)
        A single linkage tree for each cluster produced during the branch
        detection step, in the scipy hierarchical clustering format.
        (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).
        Data points are numbered with in-cluster ids.

    centralities : np.ndarray, shape (n_samples, )
        Centrality values for each point in a cluster. Overemphasizes points'
        eccentricity within the cluster as the values are based on minimum
        spanning trees that do not contain the equally distanced edges resulting
        from the mutual reachability distance.

    cluster_points : list (n_clusters)
        The data point row indices for each cluster.

    References
    ----------
    .. [1] Bot, D. M., Peeters, J., Liesenborgs J., & Aerts, J. (2023, November).
       FLASC: A Flare-Sensitive Clustering Algorithm: Extending HDBSCAN* for
       Detecting Branches in Clusters. arXiv:2311.15887
    """
    # Check clusterer state
    if clusterer._min_spanning_tree is None:
        raise ValueError(
            "Clusterer does not have an explicit minimum spanning tree!"
            " Try fitting with branch_detection_data=True or"
            " gen_min_span_tree=True set."
        )
    if clusterer.branch_detection_data_ is None:
        raise ValueError(
            "Clusterer does not have branch detection data!"
            " Try fitting with branch_detection_data=True set,"
            " or run generate_branch_detection_data on the clusterer"
        )

    # Validate parameters
    if min_branch_size is None:
        min_branch_size = clusterer.min_cluster_size
    branch_selection_epsilon = float(branch_selection_epsilon)
    branch_selection_persistence = float(branch_selection_persistence)
    if not (np.issubdtype(type(min_branch_size), np.integer) and min_branch_size >= 2):
        raise ValueError(
            f"min_branch_size must be an integer greater or equal "
            f"to 2,  {min_branch_size} given."
        )
    if not (
        np.issubdtype(type(branch_selection_persistence), np.floating)
        and branch_selection_persistence >= 0.0
    ):
        raise ValueError(
            f"branch_selection_persistence must be a float greater or equal to "
            f"0.0, {branch_selection_persistence} given."
        )
    if not (
        np.issubdtype(type(branch_selection_epsilon), np.floating)
        and branch_selection_epsilon >= 0.0
    ):
        raise ValueError(
            f"branch_selection_epsilon must be a float greater or equal to "
            f"0.0, {branch_selection_epsilon} given."
        )
    if branch_selection_method not in ("eom", "leaf"):
        raise ValueError(
            f"Invalid branch_selection_method: {branch_selection_method}\n"
            f'Should be one of: "eom", "leaf"\n'
        )
    if branch_detection_method not in ("core", "full"):
        raise ValueError(
            f"Invalid ``branch_detection_method``: {branch_detection_method}\n"
            'Should be one of: "core", "full"\n'
        )

    # Extract state
    memory = clusterer.memory
    if isinstance(memory, str):
        memory = Memory(memory, verbose=0)
    num_clusters = len(clusterer.cluster_persistence_)
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_
    if not clusterer.branch_detection_data_.all_finite:
        finite_index = clusterer.branch_detection_data_.finite_index
        labels = labels[finite_index]
        probabilities = probabilities[finite_index]

    # Configure parallelization
    run_core = branch_detection_method == "core"
    num_jobs = clusterer.core_dist_n_jobs
    if num_jobs < 1:
        num_jobs = max(cpu_count() + 1 + num_jobs, 1)
    thread_pool = (
        SequentialPool() if run_core else Parallel(n_jobs=num_jobs, max_nbytes=None)
    )

    # Detect branches
    (
        cluster_points,
        cluster_centralities,
        cluster_linkage_trees,
        cluster_approximation_graphs,
    ) = memory.cache(_compute_branch_linkage, ignore=["thread_pool"])(
        labels,
        probabilities,
        clusterer._min_spanning_tree,
        clusterer.branch_detection_data_.tree,
        clusterer.branch_detection_data_.neighbors,
        clusterer.branch_detection_data_.core_distances,
        clusterer.branch_detection_data_.dist_metric,
        num_clusters,
        thread_pool,
        run_core=run_core,
    )
    (
        branch_labels,
        branch_probabilities,
        branch_persistences,
        condensed_trees,
        linkage_trees,
    ) = memory.cache(compute_branch_segmentation, ignore=["thread_pool"])(
        linkage_trees,
        thread_pool,
        min_branch_size=min_branch_size,
        allow_single_branch=allow_single_branch,
        branch_selection_method=branch_selection_method,
        branch_selection_epsilon=branch_selection_epsilon,
        branch_selection_persistence=branch_selection_persistence,
        max_branch_size=max_branch_size,
    )
    (
        labels,
        probabilities,
        branch_labels,
        branch_probabilities,
        centralities,
    ) = memory.cache(update_labelling)(
        cluster_probabilities,
        points,
        centralities,
        branch_labels,
        branch_probabilities,
        branch_persistences,
        label_sides_as_branches=label_sides_as_branches,
    )

    # Maintain data indices for non-finite data
    if not clusterer.branch_detection_data_.all_finite:
        internal_to_raw = clusterer.branch_detection_data_.internal_to_raw
        _remap_point_lists(points, internal_to_raw)
        _remap_edge_lists(approximation_graphs, internal_to_raw)

        num_points = len(clusterer.labels_)
        labels = _remap_labels(labels, finite_index, num_points)
        probabilities = _remap_probabilities(probabilities, finite_index, num_points)
        cluster_labels = _remap_labels(cluster_labels, finite_index, num_points)
        cluster_probabilities = _remap_probabilities(
            cluster_probabilities, finite_index, num_points
        )
        branch_labels = _remap_labels(branch_labels, finite_index, num_points, 0)
        branch_probabilities = _remap_probabilities(
            branch_probabilities, finite_index, num_points
        )
        centralities = _remap_probabilities(centralities, finite_index, num_points)

    return (
        labels,
        probabilities,
        cluster_labels,
        cluster_probabilities,
        branch_labels,
        branch_probabilities,
        branch_persistences,
        approximation_graphs,
        condensed_trees,
        linkage_trees,
        centralities,
        points,
    )


def update_single_cluster_labels(
    condensed_tree,
    labels,
    probabilities,
    persistences,
    allow_single_cluster=False,
    cluster_selection_epsilon=0.0,
):
    """Sets all points up to cluster_selection_epsilon to the zero-cluster if
    a single cluster is detected."""
    if allow_single_cluster and len(persistences) == 1:
        labels = np.zeros_like(labels)
        probabilities = np.ones_like(probabilities)
        if cluster_selection_epsilon > 0.0:
            size_mask = condensed_tree["child_size"] == 1
            lambda_mask = condensed_tree["lambda_val"] < (1 / cluster_selection_epsilon)
            noise_points = condensed_tree["child"][lambda_mask & size_mask]
            labels[noise_points] = -1
            probabilities[noise_points] = 0.0

    return labels, probabilities


def compute_branch_linkage(
    cluster_labels,
    cluster_probabilities,
    min_spanning_tree,
    space_tree,
    neighbors,
    core_distances,
    dist_metric,
    num_clusters,
    thread_pool,
    run_core=False,
):
    result = thread_pool(
        delayed(_compute_branch_linkage_of_cluster)(
            cluster_labels,
            cluster_probabilities,
            min_spanning_tree,
            space_tree,
            neighbors,
            core_distances,
            dist_metric,
            run_core,
            cluster_id,
        )
        for cluster_id in range(num_clusters)
    )
    if len(result):
        return tuple(zip(*result))
    return (), (), (), ()


def _compute_branch_linkage_of_cluster(
    cluster_labels,
    cluster_probabilities,
    min_spanning_tree,
    space_tree,
    neighbors,
    core_distances,
    dist_metric,
    run_core,
    cluster_id,
):
    """Detect branches within one cluster."""
    # List points within cluster
    cluster_mask = cluster_labels == cluster_id
    cluster_points = np.where(cluster_mask)[0]
    in_cluster_ids = np.full(cluster_labels.shape[0], -1, dtype=np.double)
    in_cluster_ids[cluster_points] = np.arange(len(cluster_points), dtype=np.double)

    # Extract MST edges within cluster
    parent_mask = cluster_labels[min_spanning_tree[:, 0].astype(np.intp)] == cluster_id
    child_mask = cluster_labels[min_spanning_tree[:, 1].astype(np.intp)] == cluster_id
    cluster_mst = min_spanning_tree[parent_mask & child_mask]
    cluster_mst[:, 0] = in_cluster_ids[cluster_mst[:, 0].astype(np.intp)]
    cluster_mst[:, 1] = in_cluster_ids[cluster_mst[:, 1].astype(np.intp)]

    # Compute in cluster centrality
    points = space_tree.data.base[cluster_points]
    centroid = np.average(points, weights=cluster_probabilities[cluster_mask], axis=0)
    centralities = dist_metric.pairwise(centroid[None], points)[0, :]
    centralities = 1 / centralities

    # Construct cluster approximation graph
    if run_core:
        edges = extract_core_cluster_graph(
            cluster_mst, core_distances, neighbors[cluster_points], in_cluster_ids
        )
    else:
        max_dist = cluster_mst.T[2].max()
        edges = extract_full_cluster_graph(
            space_tree, core_distances, cluster_points, in_cluster_ids, max_dist
        )

    # Compute linkage over the graph
    return compute_branch_linkage_from_graph(
        cluster_points, centralities, edges, overridden_labels
    )


def compute_branch_linkage_from_graph(
    cluster_points, centralities, edges, overridden_labels
):
    # Set max centrality as 'distance'
    np.maximum(
        centralities[edges[:, 0].astype(np.intp)],
        centralities[edges[:, 1].astype(np.intp)],
        edges[:, 2],
    )

    # Extract MST edges
    centrality_mst = minimum_spanning_tree(
        coo_array(
            (edges[:, 2], (edges[:, 0].astype(np.int32), edges[:, 1].astype(np.int32))),
            shape=(len(cluster_points), len(cluster_points)),
        )
    ).tocoo()
    centrality_mst = np.column_stack(
        (centrality_mst.row, centrality_mst.col, centrality_mst.data)
    )
    centrality_mst = centrality_mst[np.argsort(centrality_mst.T[2]), :]
    linkage_tree = label(centrality_mst)

    # Re-label edges with data ids
    edges[:, 0] = cluster_points[edges[:, 0].astype(np.intp)]
    edges[:, 1] = cluster_points[edges[:, 1].astype(np.intp)]

    # Return values
    return cluster_points, centralities, linkage_tree, edges


def extract_core_cluster_graph(
    cluster_spanning_tree,
    core_distances,
    neighbors,
    in_cluster_ids,
):
    """Create a graph connecting all points within each point's core distance."""
    # Allocate output (won't be filled completely)
    num_points = neighbors.shape[0]
    num_neighbors = neighbors.shape[1]
    count = cluster_spanning_tree.shape[0]
    edges = np.zeros((count + num_points * num_neighbors, 4), dtype=np.double)

    # Fill (undirected) MST edges with within-cluster-ids
    mst_parents = cluster_spanning_tree[:, 0].astype(np.intp)
    mst_children = cluster_spanning_tree[:, 1].astype(np.intp)
    np.minimum(mst_parents, mst_children, edges[:count, 0])
    np.maximum(mst_parents, mst_children, edges[:count, 1])

    # Fill neighbors with within-cluster-ids
    core_parent = np.repeat(np.arange(num_points, dtype=np.double), num_neighbors)
    core_children = in_cluster_ids[neighbors.flatten()]
    np.minimum(core_parent, core_children, edges[count:, 0])
    np.maximum(core_parent, core_children, edges[count:, 1])

    # Fill mutual reachabilities
    edges[:count, 3] = cluster_spanning_tree[:, 2]
    np.maximum(
        core_distances[edges[count:, 0].astype(np.intp)],
        core_distances[edges[count:, 1].astype(np.intp)],
        edges[count:, 3],
    )

    # Extract unique edges that stay within the cluster
    edges = np.unique(edges[edges[:, 0] > -1.0, :], axis=0)
    return edges


def extract_full_cluster_graph(
    space_tree, core_distances, cluster_points, in_cluster_ids, max_dist
):
    # Query KDTree/BallTree for neighors within the distance
    children_map, distances_map = space_tree.query_radius(
        space_tree.data.base[cluster_points], r=max_dist + 1e-8, return_distance=True
    )

    # Count number of returned edges per point
    num_children = np.zeros(len(cluster_points), dtype=np.intp)
    for i, children in enumerate(children_map):
        num_children[i] += len(children)

    # Create full edge list
    full_parents = np.repeat(
        np.arange(len(cluster_points), dtype=np.double), num_children
    )
    full_children = in_cluster_ids[np.concatenate(children_map)]
    full_distances = np.concatenate(distances_map)

    # Create output
    mask = (
        (full_children != -1.0)
        & (full_parents < full_children)
        & (full_distances <= max_dist)
    )
    edges = np.zeros((mask.sum(), 4), dtype=np.double)
    edges[:, 0] = full_parents[mask]
    edges[:, 1] = full_children[mask]
    np.maximum(
        np.maximum(
            core_distances[edges[:, 0].astype(np.intp)],
            core_distances[edges[:, 1].astype(np.intp)],
        ),
        full_distances[mask],
        edges[:, 3],
    )
    return edges


def compute_branch_segmentation(
    cluster_linkage_trees,
    thread_pool,
    min_branch_size=5,
    max_branch_size=0,
    allow_single_branch=False,
    branch_selection_method="eom",
    branch_selection_epsilon=0.0,
    branch_selection_persistence=0.0,
):
    """Extracts branches from the linkage hierarchies."""
    results = thread_pool(
        delayed(_segment_branch_linkage_hierarchy)(
            cluster_linkage_tree,
            min_branch_size=min_branch_size,
            max_branch_size=max_branch_size,
            allow_single_branch=allow_single_branch,
            branch_selection_method=branch_selection_method,
            branch_selection_epsilon=branch_selection_epsilon,
            branch_selection_persistence=branch_selection_persistence,
        )
        for cluster_linkage_tree in cluster_linkage_trees
    )
    if len(results):
        return tuple(zip(*results))
    return (), (), (), (), ()


def _segment_branch_linkage_hierarchy(
    single_linkage_tree,
    min_branch_size=5,
    max_branch_size=0,
    allow_single_branch=False,
    branch_selection_method="eom",
    branch_selection_epsilon=0.0,
    branch_selection_persistence=0.0,
):
    """Select branches within one cluster."""
    # Run normal branch detection
    (labels, probabilities, stabilities, condensed_tree, linkage_tree) = (
        _tree_to_labels(
            None,
            single_linkage_tree,
            min_cluster_size=min_branch_size,
            max_cluster_size=max_branch_size,
            allow_single_cluster=allow_single_branch,
            cluster_selection_method=branch_selection_method,
            cluster_selection_epsilon=branch_selection_epsilon,
            cluster_selection_persistence=branch_selection_persistence,
        )
    )
    labels, probabilities = update_single_cluster_labels(
        condensed_tree,
        labels,
        probabilities,
        stabilities,
        allow_single_cluster=allow_single_branch,
        cluster_selection_epsilon=branch_selection_epsilon,
    )
    return (labels, probabilities, stabilities, condensed_tree, linkage_tree)


def update_labelling(
    cluster_probabilities,
    points_list,
    centrality_list,
    branch_label_list,
    branch_prob_list,
    branch_pers_list,
    label_sides_as_branches=False,
):
    """Updates the labelling with the detected branches."""
    # Allocate output
    num_points = len(cluster_probabilities)
    labels = -1 * np.ones(num_points, dtype=np.intp)
    probabilities = cluster_probabilities.copy()
    branch_labels = np.zeros(num_points, dtype=np.intp)
    branch_probabilities = np.ones(num_points, dtype=np.double)
    branch_centralities = np.zeros(num_points, dtype=np.double)

    # Compute the labels and probabilities
    running_id = 0
    for _points, _labels, _probs, _centrs, _pers in zip(
        points_list,
        branch_label_list,
        branch_prob_list,
        centrality_list,
        branch_pers_list,
    ):
        num_branches = len(_pers)
        branch_centralities[_points] = _centrs
        if num_branches <= (1 if label_sides_as_branches else 2):
            labels[_points] = running_id
            running_id += 1
        else:
            _labels[_labels == -1] = len(_pers)
            labels[_points] = _labels + running_id
            branch_labels[_points] = _labels
            branch_probabilities[_points] = _probs
            probabilities[_points] += _probs
            probabilities[_points] /= 2
            running_id += num_branches + 1

    # Reorder other parts
    return (
        labels,
        probabilities,
        branch_labels,
        branch_probabilities,
        branch_centralities,
    )


def _remap_edge_lists(edge_lists, internal_to_raw):
    """
    Takes a list of edge lists and replaces the internal indices to raw indices.

    Parameters
    ----------
    edge_lists : list[np.ndarray]
        A list of numpy edgelists with the first two columns indicating
        datapoints.
    internal_to_raw: dict
        A mapping from internal integer index to the raw integer index.
    """
    for graph in edge_lists:
        for edge in graph:
            edge[0] = internal_to_raw[edge[0]]
            edge[1] = internal_to_raw[edge[1]]


def _remap_point_lists(point_lists, internal_to_raw):
    """
    Takes a list of points lists and replaces the internal indices to raw indices.

    Parameters
    ----------
    point_lists : list[np.ndarray]
        A list of numpy arrays with point indices.
    internal_to_raw: dict
        A mapping from internal integer index to the raw integer index.
    """
    for points in point_lists:
        for idx in range(len(points)):
            points[idx] = internal_to_raw[points[idx]]


def _remap_labels(old_labels, finite_index, num_points, fill_value=-1):
    """Creates new label array with infinite points set to -1."""
    new_labels = np.full(num_points, fill_value)
    new_labels[finite_index] = old_labels
    return new_labels


def _remap_probabilities(old_probs, finite_index, num_points):
    """Creates new probability array with infinite points set to 0."""
    new_probs = np.zeros(num_points)
    new_probs[finite_index] = old_probs
    return new_probs


class BranchDetector(BaseEstimator, ClusterMixin):
    """Performs a flare-detection post-processing step to detect branches within
    clusters [1]_.

    For each cluster, a graph is constructed connecting the data points based on
    their mutual reachability distances. Each edge is given a centrality value
    based on how far it lies from the cluster's center. Then, the edges are
    clustered as if that centrality was a distance, progressively removing the
    'center' of each cluster and seeing how many branches remain.

    Parameters
    ----------
    min_branch_size : int, optional (default=None)
        The minimum number of samples in a group for that group to be
        considered a branch; groupings smaller than this size will seen as
        points falling out of a branch. Defaults to the clusterer's min_cluster_size.

    allow_single_branch : bool, optional (default=False)
        Analogous to ``allow_single_cluster``.

    branch_detection_method : str, optional (default=``full``)
        Determines which graph is constructed to detect branches with. Valid
        values are, ordered by increasing computation cost and decreasing
        sensitivity to noise:
        - ``core``: Contains the edges that connect each point to all other
          points within a mutual reachability distance lower than or equal to
          the point's core distance. This is the cluster's subgraph of the
          k-NN graph over the entire data set (with k = ``min_samples``).
        - ``full``: Contains all edges between points in each cluster with a
          mutual reachability distance lower than or equal to the distance of
          the most-distance point in each cluster. These graphs represent the
          0-dimensional simplicial complex of each cluster at the first point in
          the filtration where they contain all their points.

    branch_selection_method : str, optional (default='eom')
        The method used to select branches from the cluster's condensed tree.
        The standard approach for FLASC is to use the ``eom`` approach.
        Options are:
          * ``eom``
          * ``leaf``

    branch_selection_epsilon: float, optional (default=0.0)
        A lower epsilon threshold. Only branches with a death above this value
        will be considered.

    branch_selection_persistence: float, optional (default=0.0)
        An eccentricity persistence threshold. Branches with a persistence below
        this value will be merged.

    max_branch_size : int, optional (default=0)
        A limit to the size of clusters returned by the ``eom`` algorithm. Has
        no effect when using ``leaf`` clustering (where clusters are usually
        small regardless). Note that this should not be used if we want to
        predict the cluster labels for new points in future because
        `approximate_predict` is not aware of this argument.

    label_sides_as_branches : bool, optional (default=False),
        When this flag is False, branches are only labelled for clusters with at
        least three branches (i.e., at least y-shapes). Clusters with only two
        branches represent l-shapes. The two branches describe the cluster's
        outsides growing towards each other. Enabling this flag separates these
        branches from each other in the produced labelling.

    Attributes
    ----------
    labels_ : np.ndarray, shape (n_samples, )
        Labels that differentiate all subgroups (clusters and branches). Noisy
        samples are given the label -1.

    probabilities_ : np.ndarray, shape (n_samples, )
        Probabilities considering both cluster and branch membership. Noisy
        samples are assigned 0.

    branch_labels_ : np.ndarray, shape (n_samples, )
        Branch labels for each point. Noisy samples are given the label -1.

    branch_probabilities_ : np.ndarray, shape (n_samples, )
        Branch membership strengths for each point. Noisy samples are
        assigned 0.

    branch_persistences_ : tuple (n_clusters)
        A branch persistence (eccentricity range) for each detected branch.

    approximation_graph_ : ApproximationGraph
        The graphs used to detect branches in each cluster stored as a numpy
        array with four columns: source, target, centrality, mutual reachability
        distance. Points are labelled by their row-index into the input data.
        The edges contained in the graphs depend on the ``branch_detection_method``:
        - ``core``: Contains the edges that connect each point to all other
          points in a cluster within a mutual reachability distance lower than
          or equal to the point's core distance. This is an extension of the
          minimum spanning tree introducing only edges with equal distances. The
          reachability distance introduces ``num_points`` * ``min_samples`` of
          such edges.
        - ``full``: Contains all edges between points in each cluster with a
          mutual reachability distance lower than or equal to the distance of
          the most-distance point in each cluster. These graphs represent the
          0-dimensional simplicial complex of each cluster at the first point in
          the filtration where they contain all their points.

    condensed_trees_ : tuple (n_clusters)
        A condensed branch hierarchy for each cluster produced during the
        branch detection step. Data points are numbered with in-cluster ids.

    linkage_trees_ : tuple (n_clusters)
        A single linkage tree for each cluster produced during the branch
        detection step, in the scipy hierarchical clustering format.
        (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).
        Data points are numbered with in-cluster ids.

    centralities_ : np.ndarray, shape (n_samples, )
        Centrality values for each point in a cluster. Overemphasizes points'
        eccentricity within the cluster as the values are based on minimum
        spanning trees that do not contain the equally distanced edges resulting
        from the mutual reachability distance.

    cluster_points_ : list (n_clusters)
        The data point row indices for each cluster.

    References
    ----------
    .. [1] Bot, D. M., Peeters, J., Liesenborgs J., & Aerts, J. (2023, November).
       FLASC: A Flare-Sensitive Clustering Algorithm: Extending HDBSCAN* for
       Detecting Branches in Clusters. arXiv:2311.15887
    """

    def __init__(
        self,
        min_branch_size=None,
        allow_single_branch=False,
        branch_detection_method="full",
        branch_selection_method="eom",
        branch_selection_epsilon=0.0,
        branch_selection_persistence=0.0,
        max_branch_size=0,
        label_sides_as_branches=False,
    ):
        self.min_branch_size = min_branch_size
        self.max_branch_size = max_branch_size
        self.allow_single_branch = allow_single_branch
        self.branch_detection_method = branch_detection_method
        self.branch_selection_method = branch_selection_method
        self.branch_selection_epsilon = branch_selection_epsilon
        self.branch_selection_persistence = branch_selection_persistence
        self.label_sides_as_branches = label_sides_as_branches

        self._approximation_graphs = None
        self._condensed_trees = None
        self._cluster_linkage_trees = None
        self._branch_exemplars = None

    def fit(self, clusterer):
        """
        Perform a flare-detection post-processing step to detect branches within
        clusters.

        Parameters
        ----------
        X : HDBSCAN
            A fitted HDBSCAN object with branch detection data generated.

        Returns
        -------
        self : object
            Returns self.
        """
        self._clusterer = X
        kwargs = self.get_params()
        (
            self.labels_,
            self.probabilities_,
            self.branch_labels_,
            self.branch_probabilities_,
            self.branch_persistences_,
            self._approximation_graphs,
            self._condensed_trees,
            self._linkage_trees,
            self.centralities_,
            self.cluster_points_,
        ) = detect_branches_in_clusters(X, **kwargs)

        return self

    def fit_predict(self, X, y=None):
        """
        Perform a flare-detection post-processing step to detect branches within
        clusters [1]_.

        Parameters
        ----------
        X : HDBSCAN
            A fitted HDBSCAN object with branch detection data generated.

        Returns
        -------
        labels : ndarray, shape (n_samples, )
            subgroup labels differentiated by cluster and branch.
        """
        self.fit(X, y)
        return self.labels_

    def weighted_centroid(self, label_id, data=None):
        """Provides an approximate representative point for a given branch.
        Note that this technique assumes a euclidean metric for speed of
        computation. For more general metrics use the ``weighted_medoid`` method
        which is slower, but can work with the metric the model trained with.

        Parameters
        ----------
        label_id: int
            The id of the cluster to compute a centroid for.

        data : np.ndarray (n_samples, n_features), optional (default=None)
            A dataset to use instead of the raw data that was clustered on.

        Returns
        -------
        centroid: array of shape (n_features,)
            A representative centroid for cluster ``label_id``.
        """
        if self.labels_ is None:
            raise AttributeError("Model has not been fit to data")
        if self._clusterer._raw_data is None and data is None:
            raise AttributeError("Raw data not available")
        if label_id == -1:
            raise ValueError(
                "Cannot calculate weighted centroid for -1 cluster "
                "since it is a noise cluster"
            )
        if data is None:
            data = self._clusterer._raw_data
        mask = self.labels_ == label_id
        cluster_data = data[mask]
        cluster_membership_strengths = self.probabilities_[mask]

        return np.average(cluster_data, weights=cluster_membership_strengths, axis=0)

    def weighted_medoid(self, label_id, data=None):
        """Provides an approximate representative point for a given branch.

        Note that this technique can be very slow and memory intensive for large
        clusters. For faster results use the ``weighted_centroid`` method which
        is faster, but assumes a euclidean metric.

        Parameters
        ----------
        label_id: int
            The id of the cluster to compute a medoid for.

        data : np.ndarray (n_samples, n_features), optional (default=None)
            A dataset to use instead of the raw data that was clustered on.

        Returns
        -------
        centroid: array of shape (n_features,)
            A representative medoid for cluster ``label_id``.
        """
        if self.labels_ is None:
            raise AttributeError("Model has not been fit to data")
        if self._clusterer._raw_data is None and data is None:
            raise AttributeError("Raw data not available")
        if label_id == -1:
            raise ValueError(
                "Cannot calculate weighted centroid for -1 cluster "
                "since it is a noise cluster"
            )
        if data is None:
            data = self._clusterer._raw_data
        mask = self.labels_ == label_id
        cluster_data = data[mask]
        cluster_membership_strengths = self.probabilities_[mask]

        dist_metric = self._clusterer.branch_detection_data_.dist_metric
        dist_mat = dist_metric.pairwise(cluster_data) * cluster_membership_strengths
        medoid_index = np.argmin(dist_mat.sum(axis=1))
        return cluster_data[medoid_index]

    @property
    def approximation_graph_(self):
        """See :class:`~hdbscan.branches.BranchDetector` for documentation."""
        if self._approximation_graphs is None:
            raise AttributeError(
                "No approximation graph was generated; try running fit first."
            )
        return ApproximationGraph(
            self._approximation_graphs,
            self.labels_,
            self.probabilities_,
            self._clusterer.labels_,
            self._clusterer.probabilities_,
            self.cluster_centralities_,
            self.branch_labels_,
            self.branch_probabilities_,
            self._clusterer._raw_data,
        )

    @property
    def condensed_trees_(self):
        """See :class:`~hdbscan.branches.BranchDetector` for documentation."""
        if self._condensed_trees is None:
            raise AttributeError(
                "No condensed trees were generated; try running fit first."
            )
        return [
            (
                CondensedTree(
                    tree, self.branch_selection_method, self.allow_single_branch
                )
                if tree is not None
                else None
            )
            for tree in self._condensed_trees
        ]

    @property
    def linkage_trees_(self):
        """See :class:`~hdbscan.branches.BranchDetector` for documentation."""
        if self._linkage_trees is None:
            raise AttributeError(
                "No linkage trees were generated; try running fit first."
            )
        return [
            SingleLinkageTree(tree) if tree is not None else None
            for tree in self._linkage_trees
        ]

    @property
    def exemplars_(self):
        """See :class:`~hdbscan.branches.BranchDetector` for documentation."""
        if self._branch_exemplars is not None:
            return self._branch_exemplars
        if self._clusterer._raw_data is None:
            raise AttributeError(
                "Branch exemplars not available with precomputed " "distances."
            )
        if self._condensed_trees is None:
            raise AttributeError("No branches detected; try running fit first.")

        num_clusters = len(self._condensed_trees)
        branch_cluster_trees = [
            branch_tree[branch_tree["child_size"] > 1]
            for branch_tree in self._condensed_trees
        ]
        selected_branch_ids = [
            sorted(branch_tree._select_clusters())
            for branch_tree in self.condensed_trees_
        ]

        self._branch_exemplars = [None] * num_clusters

        for i, points in enumerate(self.cluster_points_):
            selected_branches = selected_branch_ids[i]
            if len(selected_branches) <= (1 if self.label_sides_as_branches else 2):
                continue

            self._branch_exemplars[i] = []
            raw_condensed_tree = self._condensed_trees[i]

            for branch in selected_branches:
                _branch_exemplars = np.array([], dtype=np.intp)
                for leaf in recurse_leaf_dfs(branch_cluster_trees[i], np.intp(branch)):
                    leaf_max_lambda = raw_condensed_tree["lambda_val"][
                        raw_condensed_tree["parent"] == leaf
                    ].max()
                    candidates = raw_condensed_tree["child"][
                        (raw_condensed_tree["parent"] == leaf)
                        & (raw_condensed_tree["lambda_val"] == leaf_max_lambda)
                    ]
                    _branch_exemplars = np.hstack([_branch_exemplars, candidates])
                ids = points[_branch_exemplars]
                self._branch_exemplars[i].append(self._clusterer._raw_data[ids, :])

        return self._branch_exemplars


def approximate_predict_branch(branch_detector, points_to_predict):
    """Predict the cluster and branch label of new points.

    Extends ``approximate_predict`` to also predict in which branch
    new points lie (if the cluster they are part of has branches).

    Parameters
    ----------
    branch_detector : BranchDetector
        A clustering object that has been fit to vector inpt data.

    points_to_predict : array, or array-like (n_samples, n_features)
        The new data points to predict cluster labels for. They should
        have the same dimensionality as the original dataset over which
        clusterer was fit.

    Returns
    -------
    labels : array (n_samples,)
        The predicted cluster and branch labels.

    probabilities : array (n_samples,)
        The soft cluster scores for each.

    cluster_labels : array (n_samples,)
        The predicted cluster labels.

    cluster_probabilities : array (n_samples,)
        The soft cluster scores for each.

    branch_labels : array (n_samples,)
        The predicted cluster labels.

    branch_probabilities : array (n_samples,)
        The soft cluster scores for each.
    """

    cluster_labels, cluster_probabilities, connecting_points = approximate_predict(
        branch_detector._clusterer, points_to_predict, return_connecting_points=True
    )

    num_predict = len(points_to_predict)
    labels = np.empty(num_predict, dtype=np.intp)
    probabilities = np.zeros(num_predict, dtype=np.double)
    branch_labels = np.zeros(num_predict, dtype=np.intp)
    branch_probabilities = np.ones(num_predict, dtype=np.double)

    min_num_branches = 2 if not branch_detector.label_sides_as_branches else 1
    for i, (label, prob, connecting_point) in enumerate(
        zip(cluster_labels, cluster_probabilities, connecting_points)
    ):
        if label < 0:
            labels[i] = -1
        elif len(branch_detector.branch_persistences_[label]) <= min_num_branches:
            labels[i] = label
            probabilities[i] = prob
        else:
            labels[i] = branch_detector.labels_[connecting_point]
            branch_labels[i] = branch_detector.branch_labels_[connecting_point]
            branch_probabilities[i] = branch_detector.branch_probabilities_[
                connecting_point
            ]
            probabilities[i] = (prob + branch_probabilities[i]) / 2
    return (
        labels,
        probabilities,
        cluster_labels,
        cluster_probabilities,
        branch_labels,
        branch_probabilities,
    )


class SequentialPool:
    """API of a Joblib Parallel pool but sequential execution"""

    def __init__(self):
        self.n_jobs = 1

    def __call__(self, jobs):
        return [fun(*args, **kwargs) for (fun, args, kwargs) in jobs]
