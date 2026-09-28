import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 피규어 생성
fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# 색상 정의
bg_layer = '#f0f4f8'
border_layer = '#bcccdc'
color_engine = '#ffd6e7'
color_publisher = '#c8e6c9'
color_state = '#fff59d'
color_data = '#ffe0b2'
color_ui = '#e1f5fe'

# --- 1. Subgraph (Layer) 박스 그리기 ---
# View Layer
rect_view = patches.FancyBboxPatch((5, 70), 90, 26, boxstyle="round,pad=1", facecolor=bg_layer, edgecolor=border_layer, linewidth=1.5)
ax.add_patch(rect_view)
ax.text(7, 92, "View [UI Layer (Subscriber)]", fontsize=9, fontweight='bold', color='#333333')

# Control Layer
rect_control = patches.FancyBboxPatch((5, 38), 90, 26, boxstyle="round,pad=1", facecolor=bg_layer, edgecolor=border_layer, linewidth=1.5)
ax.add_patch(rect_control)
ax.text(7, 60, "Control [Control Layer]", fontsize=9, fontweight='bold', color='#333333')

# Model Layer
rect_model = patches.FancyBboxPatch((5, 4), 90, 28, boxstyle="round,pad=1", facecolor=bg_layer, edgecolor=border_layer, linewidth=1.5)
ax.add_patch(rect_model)
ax.text(7, 28, "Model [Model Layer]", fontsize=9, fontweight='bold', color='#333333')

# --- 2. 노드(컴포넌트 박스) 그리기 ---
def draw_node(x, y, w, h, text, color, edge_color='#333333', linewidth=1.5):
    node = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", facecolor=color, edgecolor=edge_color, linewidth=linewidth)
    ax.add_patch(node)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=7.5, fontweight='bold', multialignment='center')

# UI Nodes
draw_node(10, 74, 20, 12, "Row Spinbox", color_ui)
draw_node(40, 74, 20, 12, "B-Scan Plot", color_ui)
draw_node(70, 74, 20, 12, "A-Scan Plot", color_ui)

# Control Nodes
draw_node(15, 42, 32, 12, "UltrasoundProcessorEngine", color_engine, linewidth=2)
draw_node(62, 42, 28, 12, "DataEventPublisher", color_publisher, linewidth=2)

# Model Nodes
draw_node(10, 8, 38, 15, "UIState Class\n- Chaning Status Sotrage\n- active_row/col", color_state, linewidth=2)
draw_node(52, 8, 38, 15, "UltrasoundCubeData Class\n- Invariable Mass Data Storage \n- 3D Volumes", color_data, linewidth=1.5)

# --- 3. 화살표 및 라벨 그리기 ---
# 1. UI -> Engine
ax.annotate("", xy=(25, 54), xytext=(20, 74),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.5, connectionstyle="arc3,rad=0.1"))
ax.text(24, 65, "1. User Input (Row Change: 10 -> 11)", fontsize=5, color='#111111', ha='right')

# 2. Engine -> UIState
ax.annotate("", xy=(25, 23), xytext=(25, 42),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.5))
ax.text(24, 34, "2. Value Validation State Update", fontsize=5, color='#111111', ha='right', va='center')

# 3. Engine -> UltrasoundCubeData (점선)
ax.annotate("", xy=(65, 23), xytext=(37, 42),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.5, linestyle="--"))
ax.text(54, 34, "3. Up-to-date State(Row 11) ~> Pure Data Refer (Slicing)", fontsize=5, color='#111111', ha='left', va='center')

# 4. Engine -> Publisher
ax.annotate("", xy=(62, 48), xytext=(47, 48),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.5))
ax.text(54, 52, "4. Calculation ~>\nEvent Publish (notify)", fontsize = 5, color='#111111', ha='center')

# 5. Publisher -> B-Scan Plot
ax.annotate("", xy=(50, 74), xytext=(70, 54),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.5, connectionstyle="arc3,rad=-0.15"))
ax.text(56, 68, "5. CallBack Functions work", fontsize=5, color='#111111', ha='right')

# 5. Publisher -> A-Scan Plot
ax.annotate("", xy=(80, 74), xytext=(78, 54),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.5, connectionstyle="arc3,rad=0.1"))
ax.text(82, 68, "5. CallBack Functions work", fontsize=5, color='#111111', ha='left')

plt.tight_layout()
plt.savefig("architecture_diagram.png", dpi=300, bbox_inches='tight')
plt.show()