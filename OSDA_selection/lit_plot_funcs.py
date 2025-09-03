#!/bin/env python
"""Module containing functions for plotting data for zeolite
synthesis in the lta-cage paper.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DataSeries = Sequence[float] | np.ndarray | pd.Series

CMAP = "inferno_r"
S = 30
SIZE = 100
LINEWIDTH = 1.5

# set defaults for plot sizes
BIND_LIMS = [-20, 0.1]
TEMP_LIMS = [14, 28.1]
SCS_LIMS = [0.8, 4.0]

VOL_LIMS = [50, 450]
COMP_LIMS = [0, 18]

# === Old OSDAs ===

# 4-methyl-2,3,6,7-tetrahydro-1H,5H-pyrido [3.2.1-ij] quinolinium (ITQ-29)
# "2": "C[N+]12CCCc3cccc(c31)CCC2",

# 12DM3(4MB)I, 1,2-dimethyl-3-(2-fluorobenzyl)imidazolium
# "3": "Cc1ccc(C[n+]2ccn(C)c2C)cc1",

OSDAS: dict[str, str] = {
    "1": "CC[N+](CC)(CC)CC",  # TEA
    "2": "C1COCCOCCOCCOCCOCCO1",  # 18-crown-6 ether
    "3": "C1COCCN2CCOCCOCCN(CCO1)CCOCCOCC2",  # K222
}
GRIDKWS = {"width_ratios": (0.95, 0.04), "wspace": 0.1}
# xlim and ylim for binding vs. axis 1 and axis 2 of the OSDAs from PCA
XLIM = [3, 19.1]
YLIM = [1, 11.1]
TEMPLATE_NORM = [-18, 0]
ALPHAMAP = mpl.colors.ListedColormap(
    np.array(
        [
            [0, 0, 0, 0],
        ]
    )
)


def get_literature_markers(in_literature: float) -> str:
    """Get the marker for the literature status of the OSDA.

    Args:
        in_literature (float): 1.0 if in literature, 0.0 if not.

    Returns:
        str: marker for the OSDA
    """
    if in_literature == 1.0:
        # return "^"
        pass
    return "o"


def mscatter(
    x: DataSeries,
    y: DataSeries,
    ax: mpl.axes.Axes | None = None,
    m: mmarkers.MarkerStyle | str | None = None,
    **kw: dict,
) -> mpl.collections.PathCollection:
    """Create a scatter plot with markers.

    Args:
        x (DataSeries): x-axis data
        y (DataSeries): y-axis data
        ax (plt.axes.Axes, optional): axes object. Defaults to None.
        m (mmarkers.MarkerStyle | str, optional): marker style. Defaults to None.
        kw (dict): keyword arguments for the scatter plot.

    Returns:
        plt.collections.PathCollection: scatter plot object
    """
    ax = ax or plt.gca()
    sc = ax.scatter(x, y, **kw)

    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def plot_osda_annot(
    x: DataSeries,
    y: DataSeries,
    ax: mpl.axes.Axes,
    d: pd.DataFrame,
    norm: mpl.colors.Normalize,
    osdas: dict,
    color_option: str = "Templating",
    cmap: str = CMAP,
) -> None:
    """Plot the OSDAs on the scatter plot.

    Args:
        x (DataSeries): x-axis data
        y (DataSeries): y-axis data
        ax (plt.axes.Axes): axes object
        d (pd.DataFrame): dataframe with the data
        norm (mpl.colors.Normalize): normalization object
        osdas (dict): dictionary with OSDAs, where the key is the number and the value
            is the SMILES string. See OSDAS at top for an example.
        color_option (str, optional): color option for the plot. Defaults to "Templating".
        cmap (str, optional): colormap for the plot. Defaults to CMAP.
    """
    for i, sp in osdas.items():
        sp_data = d.loc[d["SMILES"] == sp].iloc[0]
        ax.scatter(
            [sp_data[x]],
            [sp_data[y]],
            c=[sp_data[color_option]],
            s=SIZE,
            norm=norm,
            linewidths=LINEWIDTH,
            edgecolors="k",
            cmap=cmap,
            marker="s",
        )
        ax.annotate(
            i,
            (sp_data[x], sp_data[y]),
            zorder=3,
            ha="center",
            va="center",
            fontsize=8,
        )


def make_binding_templating_plot(
    df: pd.DataFrame,
    zeolite: str,
    color_option: str,
    cmap: str = "coolwarm_r",
    osdas: dict = OSDAS,
    figsize: Sequence = (2.6, 1.8),
    comp_lims: list = COMP_LIMS,
    temp_lims: list = TEMP_LIMS,
    vol_lims: list = VOL_LIMS,
    size: int = SIZE,
    mscatter_size: int = S,
    line_width: float = LINEWIDTH,
    grid_kws: dict = GRIDKWS,
) -> None:
    """This function plots binding energy and templating energy for a given zeolite
    as a function of the OSDA volume.

    Args:
        df (pd.DataFrame): pandas df with the binding energies from Schwalbe-Koda
            Science paper.
        zeolite (str): zeolite three-letter code
        color_option (str): color option for the plot, usually "Competition (SiO2)"
        cmap (str, optional): colormap for the plot. Defaults to "coolwarm_r".
        osdas (dict, optional): dictionary with OSDAs in it. Defaults to OSDAS, see for format.
        figsize (tuple, optional): size of the figure in inches. Defaults to (5, 4).
        comp_lims (tuple, optional): limits for the competition energies in kJ/mol.
            Defaults to COMP_LIMS.
        temp_lims (tuple, optional): limits for the templating energies in kJ/mol.
            Defaults to TEMP_LIMS.
        vol_lims (tuple, optional): limits for the molecular volume in Ang^3. Defaults to VOL_LIMS.
        size (int, optional): _description_. Defaults to SIZE.
        mscatter_size (_type_, optional): _description_. Defaults to S.
        line_width (_type_, optional): _description_. Defaults to LINEWIDTH.
        grid_kws (dict, optional): grid keywords for matplotlib. Defaults to GRIDKWS.
    """
    fig, ax_fig = plt.subplots(1, 2, figsize=figsize, gridspec_kw=grid_kws)

    # y1 = "Binding (SiO2)"
    y2 = "Templating"

    d = df.loc[
        (df["Zeolite"] == zeolite)
        & (df[y2] > temp_lims[0])
        & (df[y2] < temp_lims[1] - 0.1)
        & (df["Volume (Angstrom3)"] > vol_lims[0])
        & (df["Volume (Angstrom3)"] < vol_lims[1])
    ].sort_values("Templating", ascending=False)

    color_values = (d[color_option]).values.clip(min=comp_lims[0], max=comp_lims[1])  # noqa: PD011

    norm = mpl.colors.Normalize(comp_lims[0], vmax=comp_lims[1])

    if "In literature?" not in d.columns:
        d["In literature?"] = 0.0
    markers = d["In literature?"].apply(get_literature_markers).tolist()
    ax = ax_fig[0]
    x = "Volume (Angstrom3)"

    scat1 = mscatter(
        d[x],
        d[y2],
        ax=ax,
        c=color_values,
        m=markers,
        s=mscatter_size,
        norm=norm,
        linewidths=0.7,
        edgecolors="k",
        cmap=cmap,
    )

    ax.set_xlabel(
        "OSDA Volume ($\mathdefault{\AA}^{\mathdefault{3}}$)",  # noqa: W605
        fontweight="bold",
        fontsize=10,
    )
    ax.set_ylabel(
        "Templating Energy,\n$\mathdefault{E_{T}}$ (kJ mol$^{\mathdefault{\minus 1}}$)",  # noqa: W605
        fontweight="bold",
        fontsize=10,
    )
    ax.set_xlim(vol_lims)
    ax.set_ylim(temp_lims)
    ax.set_yticks(np.arange(*temp_lims, 2.0))

    for i, sp in osdas.items():
        spiro_data = d.loc[d["SMILES"] == sp].iloc[0]
        ax.scatter(
            [spiro_data[x]],
            [spiro_data[y2]],
            c=[spiro_data[color_option]],
            s=size,
            norm=norm,
            linewidths=line_width,
            edgecolors="k",
            cmap=cmap,
            marker="s",
        )
        ax.annotate(
            str(i),
            (spiro_data[x], spiro_data[y2]),
            zorder=3,
            ha="center",
            va="center",
            fontsize=8,
        )

    ax = ax_fig[1]
    cbar = fig.colorbar(scat1, cax=ax)
    cbar.set_label(
        "Competition Energy, C\n(kJ (mol SiO$_{\mathdefault{2}}$)$^{\mathdefault{\minus 1}}$)",  # noqa: W605
        fontweight="bold",
        fontsize=10,
    )
    cbar.set_ticks(np.arange(comp_lims[0], comp_lims[1] + 0.1, 2.0))

    figname = zeolite + "_binding_per_Si.svg"

    fig.savefig(
        figname,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    plt.show()


def make_axis_templating_plot(
    df: pd.DataFrame,
    zeolite: str,
    color_option: str,
    cmap="inferno_r",
    osdas=OSDAS,
    figsize=(2.6, 1.8),
    template_norm=TEMPLATE_NORM,
    temp_lims=TEMP_LIMS,
    xlim=XLIM,
    ylim=YLIM,
    grid_kws=GRIDKWS,
) -> None:
    """This function makes the plot showing templating energy on axes 1 and 2 from
    the PCA of the OSDAs.

    Args:
        df (pandas DataFrame): df with binding energies from Schwalbe-Koda
            Science paper.
        zeolite (str): zeolite name
        color_option (str): color option for the plot, usually "Templating"
        cmap (str, optional): colormap for the plot. Defaults to "inferno_r".
        osdas (dict, optional): dictionary of OSDA strings. Defaults to OSDAS, see that
            for formatting.
        figsize (tuple, optional): figure size in inches. Defaults to (5, 4).
        template_norm (tuple, optional): normalization boundaries for templating in
            colorbar in kJ/mol. Defaults to TEMPLATE_NORM.
        temp_lims (tuple, optional): limits on templating E in kJ/mol. Defaults to TEMP_LIMS.
        xlim (tuple, optional): x-axis limits. Defaults to XLIM.
        ylim (tuple, optional): y-axis limits. Defaults to YLIM.
        grid_kws (dict, optional): keywords for figure grid in matplotlib. Defaults to GRIDKWS.
    """
    norm = mpl.colors.Normalize(vmin=template_norm[0], vmax=template_norm[1])
    fig, ax_fig = plt.subplots(1, 2, figsize=figsize, gridspec_kw=grid_kws)

    ax = ax_fig[0]

    x = "Axis 1 (Angstrom)"
    y = "Axis 2 (Angstrom)"

    d = df.loc[
        (df["Zeolite"] == zeolite)
        & (df[x] > xlim[0])
        & (df[x] < xlim[1] - 0.5)
        & (df[y] > ylim[0])
        & (df[y] < ylim[1] - 0.5)
    ].sort_values("Templating", ascending=False)

    # color = d[color_option]
    # markers = d["In literature?"].apply(get_literature_markers).values.tolist()

    norm = mpl.colors.Normalize(vmin=temp_lims[0], vmax=temp_lims[1])

    scat2 = ax.hexbin(
        d[x],
        d[y],
        C=d[color_option],
        mincnt=1,
        gridsize=10,
        reduce_C_function=np.mean,
        cmap=cmap,
        norm=norm,
        extent=(xlim + ylim),
        linewidths=0.4,
        edgecolors="w",
    )

    ## Plotting the literature
    # subd = d[d["Zeolite composition?"]]

    # hb = ax.hexbin(
    #     subd[x],
    #     subd[y],
    #     gridsize=10,
    #     mincnt=1,
    #     cmap=ALPHAMAP,
    #     norm=plt.Normalize(vmin=0, vmax=1),
    #     extent=(*xlim, *ylim),
    #     linewidths=3,
    #     edgecolors="k",
    # )

    plot_osda_annot(x, y, ax, d, norm, osdas=osdas, cmap=cmap, color_option=color_option)

    ax.set_xlabel("Axis 1 ($\mathdefault{\AA}$)", fontweight="bold", fontsize=10)  # noqa: W605
    ax.set_ylabel("Axis 2 ($\mathdefault{\AA}$)", fontweight="bold", fontsize=10)  # noqa: W605
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.1, 2))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.1, 2))

    ax = ax_fig[1]
    cbar = fig.colorbar(scat2, cax=ax)  # noqa: F841
    ax.set_ylabel(
        "Templating Energy,\n$\mathdefault{E_{T}}$ (kJ mol$^{\mathdefault{\minus 1}}$)",
        fontweight="bold",
        fontsize=10,
    )

    fig.tight_layout()
    plt.show()

    figname = zeolite + "_templating_selectivity.svg"
    fig.savefig(
        figname,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
