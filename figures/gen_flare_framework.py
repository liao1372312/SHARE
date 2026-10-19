from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
PDF_PATH = OUT_DIR / "flare_framework.pdf"
SVG_PATH = OUT_DIR / "flare_framework.svg"
PNG_PATH = OUT_DIR / "flare_framework.png"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COLORS = {
    "input": "#E8F1FB",
    "graph": "#F1F5F9",
    "detect": "#FFF7E6",
    "slice": "#EAF7EA",
    "score": "#F3E8FF",
    "repair": "#FFECEC",
    "reuse": "#E6FFFB",
    "stroke": "#334155",
    "muted": "#64748B",
    "accent": "#DC2626",
    "green": "#15803D",
    "blue": "#2563EB",
}


def box(ax, xy, wh, text, fc, ec=None, lw=1.2, fontsize=8.5, weight="normal"):
    x, y = xy; w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.025", fc=fc, ec=ec or COLORS["stroke"], lw=lw)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize, weight=weight, linespacing=1.15)
    return patch


def arrow(ax, start, end, text=None, color=None, rad=0.0, style="-|>", lw=1.35, ls="-"):
    arr = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=10, lw=lw, color=color or COLORS["stroke"], linestyle=ls, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(arr)
    if text:
        mx=(start[0]+end[0])/2; my=(start[1]+end[1])/2
        ax.text(mx, my+0.035, text, ha="center", va="bottom", fontsize=7.2, color=color or COLORS["muted"])
    return arr


def draw_trace(ax):
    box(ax, (0.04,0.58), (0.15,0.22), "Multi-agent\nservice trace", COLORS["input"], weight="bold")
    agents=[("Planner","p"), ("Provider","s"), ("Executor","e"), ("Supervisor","r"), ("Summarizer","c")]
    xs=[0.065,0.095,0.125,0.155,0.185]
    ys=[0.49,0.42,0.35,0.28,0.21]
    prev=None
    for (label,prefix),x,y in zip(agents,xs,ys):
        c=Circle((x,y),0.018,fc="white",ec=COLORS["blue"],lw=1.1)
        ax.add_patch(c)
        ax.text(x,y,prefix,ha="center",va="center",fontsize=7,color=COLORS["blue"],weight="bold")
        ax.text(x+0.028,y,label,ha="left",va="center",fontsize=6.8,color=COLORS["muted"])
        if prev: arrow(ax, prev, (x-0.018,y), color=COLORS["muted"], lw=0.9, style="->")
        prev=(x+0.018,y)


def draw_graph(ax):
    box(ax, (0.245,0.56), (0.16,0.24), "Causal execution\ngraph construction", COLORS["graph"], weight="bold")
    nodes=[(0.28,0.45,"p"),(0.33,0.45,"s"),(0.38,0.45,"b"),(0.305,0.35,"e"),(0.365,0.35,"o"),(0.335,0.25,"c")]
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(1,3),(2,4)]
    for i,j in edges: arrow(ax,(nodes[i][0]+0.014,nodes[i][1]),(nodes[j][0]-0.014,nodes[j][1]),color="#94A3B8",lw=0.8,style="->")
    for x,y,t in nodes:
        circ=Circle((x,y),0.018,fc="white",ec=COLORS["stroke"],lw=1)
        ax.add_patch(circ); ax.text(x,y,t,ha="center",va="center",fontsize=7)


def main():
    fig, ax = plt.subplots(figsize=(8.8, 4.55))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

    # main modules
    draw_trace(ax)
    draw_graph(ax)
    box(ax,(0.45,0.62),(0.14,0.18),"Failure node\ndetection",COLORS["detect"],weight="bold")
    box(ax,(0.45,0.33),(0.14,0.18),"Backward\nexecution slicing",COLORS["slice"],weight="bold")
    box(ax,(0.64,0.48),(0.17,0.24),"Faulty decision\nlocalization",COLORS["score"],weight="bold")
    ax.text(0.725,0.535,"Error + Impact\n+ Proximity\n+ Violation\n− RepairCost",ha="center",va="center",fontsize=7.2,color="#4C1D95",linespacing=1.12)
    box(ax,(0.86,0.58),(0.12,0.16),"Minimal\nrepair action",COLORS["repair"],weight="bold")
    box(ax,(0.86,0.32),(0.12,0.16),"Partial\nre-execution",COLORS["reuse"],weight="bold")
    box(ax,(0.66,0.12),(0.18,0.13),"Repaired answer\nwith reused evidence", "#F8FAFC", weight="bold")

    # arrows
    arrow(ax,(0.19,0.69),(0.245,0.69),"events")
    arrow(ax,(0.405,0.68),(0.45,0.70),"failure signals")
    arrow(ax,(0.52,0.62),(0.52,0.51),"failures",rad=0.0)
    arrow(ax,(0.59,0.42),(0.64,0.56),"slice")
    arrow(ax,(0.81,0.60),(0.86,0.66),"top-1 root cause")
    arrow(ax,(0.92,0.58),(0.92,0.48),"invalidate")
    arrow(ax,(0.86,0.40),(0.84,0.22),"new observations",rad=0.05)
    arrow(ax,(0.66,0.18),(0.18,0.22),"feedback to trace",color=COLORS["green"],rad=-0.22,ls="--",lw=1.1)

    # callouts
    box(ax,(0.625,0.80),(0.22,0.10),"Decision-level repair target\n(not just agent attribution)","#FFFFFF",ec=COLORS["accent"],lw=1.2,fontsize=7.5)
    arrow(ax,(0.72,0.80),(0.72,0.72),color=COLORS["accent"],lw=1.0)
    box(ax,(0.04,0.06),(0.31,0.10),"Key output: localized faulty decision,\naffected downstream slice, and repair plan", "#FFFFFF", ec="#CBD5E1", fontsize=7.5)
    box(ax,(0.39,0.06),(0.21,0.10),"Cost control:\nreuse unaffected nodes", "#FFFFFF", ec="#CBD5E1", fontsize=7.5)

    fig.savefig(PDF_PATH)
    fig.savefig(SVG_PATH)
    fig.savefig(PNG_PATH,dpi=300)
    plt.close(fig)
    print(f"Saved {PDF_PATH}")
    print(f"Saved {SVG_PATH}")

if __name__ == "__main__":
    main()
