import itertools as it
from typing import Tuple, Union
import numpy as np
import scipy.stats as ss
from statsmodels.sandbox.stats.multicomp import multipletests
from pandas import DataFrame


def __convert_to_df(
        a: Union[list, np.ndarray, DataFrame],
        val_col: str = 'vals',
        group_col: str = 'groups',
        val_id: int = None,
        group_id: int = None) -> Tuple[DataFrame, str, str]:
    '''Hidden helper method to create a DataFrame with input data for further
    processing.

    Parameters
    ----------
    a : Union[list, np.ndarray, DataFrame]
        An array, any object exposing array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary, i.e. groups
        may have different lengths.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values
        (test or response variable). Values should have a non-nominal scale.
        Must be specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    val_id : int, optional
        Index of a column that contains dependent variable values (test or
        response variable). Should be specified if a NumPy ndarray is used as an
        input. It will be inferred from data, if not specified.

    group_id : int, optional
        Index of a column that contains independent variable values (grouping or
        predictor variable). Should be specified if a NumPy ndarray is used as
        an input. It will be inferred from data, if not specified.

    Returns
    -------
    Tuple[DataFrame, str, str]
        Returns a tuple of DataFrame and two strings:
        - DataFrame with input data, `val_col` column contains numerical values
          and `group_col` column contains categorical values.
        - Name of a DataFrame column that contains dependent variable values
          (test or response variable).
        - Name of a DataFrame column that contains independent variable values
          (grouping or predictor variable).

    Notes
    -----
    Inferrence algorithm for determining `val_id` and `group_id` args is rather
    simple, so it is better to specify them explicitly to prevent errors.
    '''
    if not group_col:
        group_col = 'groups'
    if not val_col:
        val_col = 'vals'

    if isinstance(a, DataFrame):
        x = a.copy()
        if not {group_col, val_col}.issubset(a.columns):
            raise ValueError('Specify correct column names using `group_col` and `val_col` args')
        return x, val_col, group_col

    elif isinstance(a, list) or (isinstance(a, np.ndarray) and not a.shape.count(2)):
        grps_len = map(len, a)
        grps = list(it.chain(*[[i+1] * l for i, l in enumerate(grps_len)]))
        vals = list(it.chain(*a))

        return DataFrame({val_col: vals, group_col: grps}), val_col, group_col

    elif isinstance(a, np.ndarray):

        # cols ids not defined
        # trying to infer
        if not all([val_id, group_id]):

            if np.argmax(a.shape):
                a = a.T

            ax = [np.unique(a[:, 0]).size, np.unique(a[:, 1]).size]

            if np.diff(ax).item():
                __val_col = np.argmax(ax)
                __group_col = np.argmin(ax)
            else:
                raise ValueError('Cannot infer input format.\nPlease specify `val_id` and `group_id` args')

            cols = {__val_col: val_col,
                    __group_col: group_col}
        else:
            cols = {val_id: val_col,
                    group_id: group_col}

        cols_vals = dict(sorted(cols.items())).values()
        return DataFrame(a, columns=cols_vals), val_col, group_col


def __convert_to_block_df(
        a,
        y_col: str = None,
        group_col: str = None,
        block_col: str = None,
        melted: bool = False) -> DataFrame:
    # TODO: refactor conversion of block data to DataFrame
    if melted and not all([i is not None for i in [block_col, group_col, y_col]]):
        raise ValueError('`block_col`, `group_col`, `y_col` should be explicitly specified if using melted data')

    if isinstance(a, DataFrame) and not melted:
        x = a.copy(deep=True)
        group_col = 'groups'
        block_col = 'blocks'
        y_col = 'y'
        x.columns.name = group_col
        x.index.name = block_col
        x = x.reset_index().melt(id_vars=block_col, var_name=group_col, value_name=y_col)

    elif isinstance(a, DataFrame) and melted:
        x = DataFrame.from_dict({'groups': a[group_col],
                                 'blocks': a[block_col],
                                 'y': a[y_col]})

    elif not isinstance(a, DataFrame):
        x = np.array(a)
        x = DataFrame(x, index=np.arange(x.shape[0]), columns=np.arange(x.shape[1]))

        if not melted:
            group_col = 'groups'
            block_col = 'blocks'
            y_col = 'y'
            x.columns.name = group_col
            x.index.name = block_col
            x = x.reset_index().melt(id_vars=block_col, var_name=group_col, value_name=y_col)

        else:
            x.rename(columns={group_col: 'groups', block_col: 'blocks', y_col: 'y'}, inplace=True)

    group_col = 'groups'
    block_col = 'blocks'
    y_col = 'y'

    return x, y_col, group_col, block_col

def my_posthoc_dunn(
        a: Union[list, np.ndarray, DataFrame],
        val_col: str = None,
        group_col: str = None,
        p_adjust: str = None,
        sort: bool = True) -> DataFrame:
    '''Post hoc pairwise test for multiple comparisons of mean rank sums
    (Dunn's test). May be used after Kruskal-Wallis one-way analysis of
    variance by ranks to do pairwise comparisons [1]_, [2]_.

    Parameters
    ----------
    a : array_like or pandas DataFrame object
        An array, any object exposing the array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary,
        i.e. groups may have different lengths.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values (test
        or response variable). Values should have a non-nominal scale. Must be
        specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    p_adjust : str, optional
        Method for adjusting p values. See `statsmodels.sandbox.stats.multicomp`
        for details. Available methods are:
        'bonferroni' : one-step correction
        'sidak' : one-step correction
        'holm-sidak' : step-down method using Sidak adjustments
        'holm' : step-down method using Bonferroni adjustments
        'simes-hochberg' : step-up method  (independent)
        'hommel' : closed method based on Simes tests (non-negative)
        'fdr_bh' : Benjamini/Hochberg  (non-negative)
        'fdr_by' : Benjamini/Yekutieli (negative)
        'fdr_tsbh' : two stage fdr correction (non-negative)
        'fdr_tsbky' : two stage fdr correction (non-negative)

    sort : bool, optional
        Specifies whether to sort DataFrame by group_col or not. Recommended
        unless you sort your data manually.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    A tie correction will be employed according to Glantz (2012).

    References
    ----------
    .. [1] O.J. Dunn (1964). Multiple comparisons using rank sums.
        Technometrics, 6, 241-252.
    .. [2] S.A. Glantz (2012), Primer of Biostatistics. New York: McGraw Hill.

    Examples
    --------

    >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
    >>> sp.posthoc_dunn(x, p_adjust = 'holm')
    '''
    def compare_dunn(i, j):
        diff = np.abs(x_ranks_avg.loc[i] - x_ranks_avg.loc[j])
        A = n * (n + 1.) / 12.
        B = (1. / x_lens.loc[i] + 1. / x_lens.loc[j])
        z_value = diff / np.sqrt((A - x_ties) * B)
        p_value = 2. * ss.norm.sf(np.abs(z_value))
        effect  = (z_value**2)/n
        
        return p_value, z_value, effect

    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col, _val_col], ascending=True) if sort else x

    n = len(x.index)
    x_groups_unique = x[_group_col].unique()
    x_len = x_groups_unique.size
    x_lens = x.groupby(_group_col)[_val_col].count()

    x['ranks'] = x[_val_col].rank()
    x_ranks_avg = x.groupby(_group_col)['ranks'].mean()

    # ties
    vals = x.groupby('ranks').count()[_val_col].values
    tie_sum = np.sum(vals[vals != 1] ** 3 - vals[vals != 1])
    tie_sum = 0 if not tie_sum else tie_sum
    x_ties = tie_sum / (12. * (n - 1))

    vs    = np.zeros((x_len, x_len))
    zval  = np.zeros((x_len, x_len))
    ef    = np.zeros((x_len, x_len))
    
    combs = it.combinations(range(x_len), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0

    for i, j in combs:
        p, z, effect = compare_dunn(x_groups_unique[i], x_groups_unique[j])
        vs[i, j]   = p
        zval[i, j] = z
        ef[i, j]   = effect

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower]   = np.transpose(vs)[tri_lower]
    zval[tri_lower] = np.transpose(zval)[tri_lower]
    ef[tri_lower]   = np.transpose(ef)[tri_lower]
    
    np.fill_diagonal(vs, 1)
    np.fill_diagonal(ef, np.nan)
    
    df_vs = DataFrame(vs, index=x_groups_unique, columns=x_groups_unique)
    df_z  = DataFrame(zval, index=x_groups_unique, columns=x_groups_unique)
    df_ef = DataFrame(ef, index=x_groups_unique, columns=x_groups_unique)
    
    return df_vs, df_z, df_ef