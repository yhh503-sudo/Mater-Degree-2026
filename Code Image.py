import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=100)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Layer Containers (Clusters)
    # 1. View Layer
    view_bg = patches.FancyBboxPatch((3, 68), 94, 28, boxstyle="round,pad=1", fc="#f0f4f8", ec="#b0bec5", lw=2)
    ax.add_patch(view_bg)
    ax.text(5, 92, "View [UI Layer (Subscriber)]", fontsize=14, weight='bold', color="#212121")

    # 2. Control Layer
    ctrl_bg = patches.FancyBboxPatch((3, 38), 94, 26, boxstyle="round,pad=1", fc="#f0f4f8", ec="#b0bec5", lw=2)
    ax.add_patch(ctrl_bg)
    ax.text(5, 60, "Control [Control Layer]", fontsize=14, weight='bold', color="#212121")

    # 3. Model Layer
    model_bg = patches.FancyBboxPatch((3, 3), 94, 31, boxstyle="round,pad=1", fc="#f0f4f8", ec="#b0bec5", lw=2)
    ax.add_patch(model_bg)
    ax.text(5, 30, "Model [Model Layer]", fontsize=14, weight='bold', color="#212121")

    # Nodes (Boxes)
    def draw_box(x, y, w, h, text, fc, ec):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc=fc, ec=ec, lw=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, weight='bold', color="#212121")

    # View Nodes
    draw_box(8, 73, 24, 14, "Row Spinbox\n(Input Event)", "#ffffff", "#263238")
    draw_box(40, 73, 24, 14, "B-Scan Plot\n(Render Engine)", "#e1f5fe", "#0288d1")
    draw_box(70, 73, 24, 14, "A-Scan Plot\n(Render Engine)", "#e1f5fe", "#0288d1")

    # Control Nodes
    draw_box(12, 43, 36, 12, "UltrasoundProcessorEngine\n- Validates Input\n- Updates State & Slice Data", "#ffe0b2", "#f57c00")
    draw_box(60, 43, 32, 12, "DataEventPublisher\n- Manages Event Subscribers\n- Notifies Views", "#c8e6c9", "#388e3c")

    # Model Nodes
    draw_box(8, 8, 40, 16, "AppState Class\n- Dynamic Status Storage\n- active_row / active_col / configs", "#fff9c4", "#fbc02d")
    draw_box(54, 8, 40, 16, "UltrasoundCubeData Class\n- Invariable Mass Data Storage\n- 3D Volumes (Raw / Filtered)", "#ffcc80", "#e65100")

    # Arrows & Labels
    # 1. User Input
    ax.annotate("", xy=(20, 55), xytext=(20, 73), arrowprops=dict(arrowstyle="->", lw=2, color="#212121"))
    ax.text(21, 64, "1. User Input (Row Change: 10 -> 11)", fontsize=9, color="#212121")

    # 2. Value Validation State Update
    ax.annotate("", xy=(28, 24), xytext=(28, 43), arrowprops=dict(arrowstyle="->", lw=2, color="#d32f2f"))
    ax.text(29, 34, "2. Value Validation State Update", fontsize=9, color="#d32f2f", weight='bold')

    # 3. Up-to-date State Reference
    ax.annotate("", xy=(74, 24), xytext=(48, 43), arrowprops=dict(arrowstyle="->", lw=2, linestyle="--", color="#e65100"))
    ax.text(56, 34, "3. Up-to-date State (Row 11)\n~> Pure Data Refer (Slicing)", fontsize=9, color="#e65100")

    # 4. Calculation ~> Event Publish
    ax.annotate("", xy=(60, 49), xytext=(48, 49), arrowprops=dict(arrowstyle="->", lw=2, color="#388e3c"))
    ax.text(48.5, 51, "4. Calculation ~>\nEvent Publish (notify)", fontsize=8, color="#388e3c")

    # 5. CallBack Functions work
    ax.annotate("", xy=(52, 73), xytext=(72, 55), arrowprops=dict(arrowstyle="->", lw=2, color="#0288d1"))
    ax.annotate("", xy=(82, 73), xytext=(78, 55), arrowprops=dict(arrowstyle="->", lw=2, color="#0288d1"))
    ax.text(50, 62, "5. CallBack Functions work", fontsize=9, color="#0288d1")
    ax.text(80, 62, "5. CallBack Functions work", fontsize=9, color="#0288d1")

    plt.tight_layout()
    plt.savefig('architecture_diagram_matplotlib.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    generate_architecture_diagram()