import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math
import warnings
import io
import os
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

warnings.filterwarnings("ignore")

# ===================== 全局字体配置（与迈克尔逊项目完全对齐，云端+本地双兼容） =====================
FONT_FILE = "SourceHanSansCN-Regular1.otf"
try:
    # 云端：加载项目内字体并全局注册
    fm.fontManager.addfont(FONT_FILE)
    plt.rcParams["font.sans-serif"] = ["Source Han Sans CN"]
except Exception:
    # 本地降级兼容
    plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]

plt.rcParams["axes.unicode_minus"] = False
plt.switch_backend("Agg")
# ==================================================================================================

MAX_USES_PER_HOUR = 3


# ===================== 导出次数限制逻辑 =====================
def check_usage_limit(mode: str) -> bool:
    now = datetime.now()
    key = f"usage_records_{mode}"
    if key not in st.session_state:
        st.session_state[key] = []
    valid_records = [t for t in st.session_state[key] if now - t < timedelta(hours=1)]
    st.session_state[key] = valid_records
    return len(valid_records) < MAX_USES_PER_HOUR


def add_usage_record(mode: str) -> None:
    key = f"usage_records_{mode}"
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].append(datetime.now())


def get_remaining_uses(mode: str) -> int:
    now = datetime.now()
    key = f"usage_records_{mode}"
    if key not in st.session_state:
        return MAX_USES_PER_HOUR
    valid_records = [t for t in st.session_state[key] if now - t < timedelta(hours=1)]
    return max(0, MAX_USES_PER_HOUR - len(valid_records))


# ===================== 图片水印（三层降级方案，彻底解决方框问题） =====================
def add_image_watermark(img: Image.Image, p_angle: int, a_angle: int) -> Image.Image:
    draw = ImageDraw.Draw(img)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    water_text = f"{now_str}\n起偏器：{p_angle}°  检偏器：{a_angle}°"

    # 三层降级：和迈克尔逊项目完全对齐
    try:
        if os.path.exists(FONT_FILE):
            font = ImageFont.truetype(FONT_FILE, 22)
        else:
            raise FileNotFoundError("项目内字体文件不存在")
    except Exception:
        try:
            # 尝试加载云端预装的文泉驿字体
            font = ImageFont.truetype("WenQuanYi Zen Hei", 22)
        except Exception:
            # 终极兜底
            font = ImageFont.load_default(size=22)

    draw.text((15, img.height - 60), water_text, fill=(255, 0, 0), font=font)
    return img


# ===================== 偏振光绘图逻辑 =====================
def draw_rotated_line(ax, x_center, y_center, base_length, angle_deg, color='red', linewidth=2, is_analyzed=False,
                      analyzer_angle=0):
    angle_rad = math.radians(angle_deg)
    if is_analyzed:
        angle_diff = abs(angle_deg - analyzer_angle) % 180
        intensity_factor = abs(math.cos(math.radians(angle_diff)))
        current_length = base_length * intensity_factor
    else:
        current_length = base_length
    if current_length <= 0.01:
        return
    half_length = current_length / 2
    x_start = x_center - half_length * math.sin(angle_rad)
    y_start = y_center - half_length * math.cos(angle_rad)
    x_end = x_center + half_length * math.sin(angle_rad)
    y_end = y_center + half_length * math.cos(angle_rad)
    ax.plot([x_start, x_end], [y_start, y_end], linestyle='-', color=color, linewidth=linewidth)


def calculate_transmittance(polarizer_angle, analyzer_angle):
    angle_diff = abs(polarizer_angle - analyzer_angle) % 180
    return math.cos(math.radians(angle_diff)) ** 2


def calculate_measured_transmittance(polarizer_angle, analyzer_angle, bg):
    theory = calculate_transmittance(polarizer_angle, analyzer_angle)
    measured = (theory + bg) / (1 + bg)
    return measured


def plot_polarization_experiment(polarizer_angle, analyzer_angle, bg):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.plot([-27, 23], [0, 0], linestyle='--', color='black', linewidth=1)
    for x_pos in [-25, -23, -21]:
        ax.plot([x_pos, x_pos], [-3, 3], linestyle='-', color='red', linewidth=2)
        ax.plot(x_pos, 0, 'o', markersize=6, color='red')

    import matplotlib.patches as patches
    ellipse1 = patches.Ellipse((-12, 0), width=7, height=12, angle=0, edgecolor='black', facecolor='green', alpha=0.3,
                               linewidth=2)
    ax.add_patch(ellipse1)
    ellipse2 = patches.Ellipse((7, 0), width=7, height=12, angle=0, edgecolor='purple', facecolor='green', alpha=0.3,
                               linewidth=2)
    ax.add_patch(ellipse2)
    rectangle = patches.Rectangle((23, -5), width=5, height=10, edgecolor='black', facecolor='white', alpha=0.5,
                                  linewidth=2)
    ax.add_patch(rectangle)

    transmittance = calculate_measured_transmittance(polarizer_angle, analyzer_angle, bg)
    if transmittance > 0.01:
        circle = patches.Circle((25.5, 0), radius=0.5, edgecolor='blue', facecolor='red',
                                alpha=0.1 + 0.9 * transmittance, linewidth=0.1)
        ax.add_patch(circle)

    draw_rotated_line(ax, -12, 0, 7, polarizer_angle, 'green', 2, is_analyzed=False)
    for x_pos in [-5, -3, -1]:
        draw_rotated_line(ax, x_pos, 0, 6, polarizer_angle, 'red', 2, is_analyzed=False)
    draw_rotated_line(ax, 7, 0, 7, analyzer_angle, 'green', 2, is_analyzed=False)
    for x_pos in [14, 16, 18]:
        draw_rotated_line(ax, x_pos, 0, 6, analyzer_angle, 'red', 2, is_analyzed=True, analyzer_angle=polarizer_angle)

    ax.set_xlim(-30, 30)
    ax.set_ylim(-12, 12)
    ax.set_aspect('equal')
    ax.set_facecolor('#f8f9fa')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('#666')

    # 全局字体已生效，无需额外指定参数
    ax.text(-23, 8, '自然光', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    ax.text(-12, 8, f'起偏器\n{polarizer_angle}°', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    ax.text(7, 8, f'检偏器\n{analyzer_angle}°', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    ax.text(25.5, 8, f'光 屏\n {transmittance * 100:.1f}%', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    return fig


# ===================== 马吕斯动态绘图 =====================
def plot_malus_scatter(current_angle, bg):
    import random
    noisy_angle = current_angle + random.uniform(-1, 1)
    current_theory = calculate_transmittance(0, noisy_angle)
    current_measured = (current_theory + bg) / (1 + bg)

    filtered_data = [
        (ang, val) for ang, val in st.session_state.collected_malus_data
        if ang <= current_angle
    ]
    collected_dict = {ang: val for ang, val in filtered_data}
    collected_dict[current_angle] = current_measured
    st.session_state.collected_malus_data = list(collected_dict.items())

    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
    theta = np.linspace(90, 270, 200)
    theory_I = np.cos(np.radians(theta)) ** 2

    ax.plot(theta, theory_I, color='blue', linestyle='--', linewidth=1, alpha=0.6, label='理论曲线')
    collected = st.session_state.collected_malus_data
    if collected:
        collected_sorted = sorted(collected, key=lambda x: x[0])
        ang_list = [item[0] for item in collected_sorted]
        val_list = [item[1] for item in collected_sorted]
        ax.plot(ang_list, val_list, color='red', linestyle='-', linewidth=1, alpha=0.8, label='实测曲线（模拟）')

    current_theory_true = calculate_transmittance(0, current_angle)
    current_measured_true = (current_theory_true + bg) / (1 + bg)
    ax.scatter(current_angle, current_measured_true, color='red', s=50, zorder=5)

    ax.set_title(r"$\theta$ - I 线性关系", fontsize=10)
    ax.set_ylabel("归一化光强", fontsize=8)
    ax.set_xlim(90, 270)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=7)
    ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


# ===================== 实验数据预览弹窗 =====================
@st.dialog("实验数据预览", width="medium")
def show_malus_data(bg):
    st.success("✅ 所有数据已采集完成！")
    angles = list(range(90, 271, 10))
    theory_list = []
    measured_list = []

    for ang in angles:
        theory = math.cos(math.radians(ang)) ** 2
        measured = (theory + bg) / (1 + bg)
        theory_list.append(round(theory, 3))
        measured_list.append(round(measured, 3))

    columns = ["角度/°"] + [str(a) for a in angles]
    theory_row = ["理论值"] + [f"{v:.3f}" for v in theory_list]
    measured_row = ["实测值"] + [f"{v:.3f}" for v in measured_list]
    df = pd.DataFrame([theory_row, measured_row], columns=columns)
    st.dataframe(df, use_container_width=True, hide_index=True)

    collected = st.session_state.collected_malus_data
    if collected:
        x_data = []
        y_data = []
        for ang, i in collected:
            cos2_theta = math.cos(math.radians(ang)) ** 2
            x_data.append(cos2_theta)
            y_data.append(i)

        slope, intercept = np.polyfit(x_data, y_data, 1)
        y_fit = [slope * x + intercept for x in x_data]

        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        ax.scatter(x_data, y_data, color='red', label='实测数据点')
        ax.plot(x_data, y_fit, color='blue', linestyle='-', label=f'拟合直线: y={slope:.3f}x+{intercept:.3f}')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='理想直线: y=x')

        ax.set_title(r"$\cos^2\theta$ - I 线性关系", fontsize=10)
        ax.set_xlabel(r"$\cos^2\theta$", fontsize=8)
        ax.set_ylabel("归一化光强 I", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(f"""
            > **误差说明：**
            > - 拟合直线截距 `{intercept:.3f}` 对应实验环境背景噪声
            > - 斜率 `{slope:.3f}` 与理论值 1 的偏差来自角度转动误差
            > - 扣除背景噪声后，数据与马吕斯定律线性关系吻合
        """)


# ===================== 弹窗：实验原理 & 使用说明 =====================
@st.dialog("实验原理", width="small")
def show_principle():
    st.markdown(
        """<div style="font-size:17px; line-height:1.6;">1. 自然光通过起偏器后成为线偏振光。<br>2. 转动检偏器，光强随角度周期性变化。<br>3. 0°最亮，90°消光。<br>4. 马吕斯定律：I=I'×cos²θ</div>""",
        unsafe_allow_html=True)


@st.dialog("使用说明", width="small")
def show_guide():
    st.markdown(
        """<div style="font-size:17px; line-height:1.6;"><b>操作步骤</b><br>1. 选择偏振光产生和检验/马吕斯定律<br>2. 调整起偏器、检偏器角度<br>3. 观察光屏亮度<br>4. 偏振光产生和检验模式可导出带水印PNG，3次/小时</div>""",
        unsafe_allow_html=True)


# ===================== 主程序 =====================
def main():
    st.set_page_config(page_title="偏振光实验", page_icon="🔬", layout="wide")
    st.markdown("""
    <style>
    header[data-testid="stHeader"] {display:none !important}
    div[data-testid="stToolbar"] {display:none !important}
    .block-container {
    padding-top: 1px !important;
    padding-left: 8px !important;
    padding-right: 8px !important;
    max-width: 100% !important;
}
@media (min-width: 769px) {
    .block-container {
        padding-top: 100px !important;
    }
}
    html, body { font-size: 15px !important; }
    .stButton>button { font-size: 15px !important; }
    @media (max-width:768px){
        .stPyplot img { max-height: 55vh !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    mobile_view = False
    try:
        headers = st.context.headers
        user_agent = headers.get('User-Agent', '').lower()
        mobile_view = 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent
    except Exception:
        mobile_view = False

    # 初始化会话状态
    if "polarizer_angle" not in st.session_state:
        st.session_state.polarizer_angle = 0
    if "analyzer_angle" not in st.session_state:
        st.session_state.analyzer_angle = 90
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0
    if "mode" not in st.session_state:
        st.session_state.mode = "demo"
    if "bg_noise" not in st.session_state:
        st.session_state.bg_noise = 0.1
    if "collected_malus_data" not in st.session_state:
        st.session_state.collected_malus_data = []

    # 手机端布局
    if mobile_view:
        if st.button("实验原理", use_container_width=True):
            show_principle()
        if st.button("使用说明", use_container_width=True):
            show_guide()

        bg = st.session_state.bg_noise if st.session_state.mode == "sim" else 0.0
        fig_main = plot_polarization_experiment(st.session_state.polarizer_angle, st.session_state.analyzer_angle, bg)
        st.pyplot(fig_main, use_container_width=True)
        plt.close(fig_main)

        if st.button("重置", use_container_width=True):
            st.session_state.polarizer_angle = 0
            st.session_state.analyzer_angle = 90
            st.session_state.bg_noise = 0.1
            st.session_state.collected_malus_data = []
            st.session_state.reset_counter += 1
            st.rerun()

        if st.button("偏振光产生和检验", type="primary" if st.session_state.mode == "demo" else "secondary",
                     use_container_width=True):
            st.session_state.mode = "demo"
            st.rerun()
        if st.button("马吕斯定律", type="primary" if st.session_state.mode == "sim" else "secondary",
                     use_container_width=True):
            st.session_state.mode = "sim"
            st.session_state.analyzer_angle = 90
            st.session_state.collected_malus_data = []
            st.rerun()

        if st.session_state.mode == "demo":
            col_label, col_input = st.columns([1, 1])
            with col_label: st.markdown("起偏器角度")
            with col_input:
                st.session_state.polarizer_angle = st.number_input("起偏器角度", 0, 360,
                                                                   st.session_state.polarizer_angle, 10, "%d",
                                                                   label_visibility="collapsed",
                                                                   key=f"p{st.session_state.reset_counter}")

        if st.session_state.mode == "sim":
            col_label, col_input = st.columns([1, 1])
            with col_label: st.markdown("环境光噪声")
            with col_input:
                st.session_state.bg_noise = st.number_input("环境光噪声", min_value=0.10, max_value=0.30,
                                                            value=st.session_state.bg_noise, step=0.02, format="%.2f",
                                                            label_visibility="collapsed",
                                                            key=f"bg{st.session_state.reset_counter}")

        col_label, col_input = st.columns([1, 1])
        with col_label:
            st.markdown("检偏器角度")
        with col_input:
            if st.session_state.mode == "demo":
                st.session_state.analyzer_angle = st.number_input("检偏器角度", 0, 360, st.session_state.analyzer_angle,
                                                                  10, "%d", label_visibility="collapsed",
                                                                  key=f"a{st.session_state.reset_counter}")
            else:
                st.session_state.analyzer_angle = st.number_input("检偏器角度_malus", 90, 270,
                                                                  st.session_state.analyzer_angle, 10, "%d",
                                                                  label_visibility="collapsed",
                                                                  key=f"am{st.session_state.reset_counter}")

        if st.session_state.mode == "demo":
            st.markdown("演示结果导出")
            remain = get_remaining_uses("demo_export")
            disabled = remain <= 0
            col_export, col_dl = st.columns(2)
            with col_export:
                label = f"导出PNG（{remain}次）" if remain > 0 else "导出PNG（已用完）"
                if st.button(label, use_container_width=True, disabled=disabled):
                    if check_usage_limit("demo_export"):
                        add_usage_record("demo_export")
                        fig = plot_polarization_experiment(st.session_state.polarizer_angle,
                                                           st.session_state.analyzer_angle, 0.0)
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                        buf.seek(0)
                        img = Image.open(buf).convert("RGB")
                        img = add_image_watermark(img, st.session_state.polarizer_angle,
                                                  st.session_state.analyzer_angle)
                        out_buf = io.BytesIO()
                        img.save(out_buf, format="PNG")
                        st.session_state.export_png = out_buf.getvalue()
                        plt.close(fig)
                        st.success("✅ 图片已添加水印并生成")
                        st.rerun()
            with col_dl:
                if "export_png" in st.session_state:
                    st.download_button("⬇️ 下载PNG", st.session_state.export_png, "偏振光实验图.png", "image/png",
                                       use_container_width=True)

        if st.session_state.mode == "sim":
            fig_scat = plot_malus_scatter(st.session_state.analyzer_angle, st.session_state.bg_noise)
            st.pyplot(fig_scat, use_container_width=True)
            plt.close(fig_scat)

            total_required = len(list(range(90, 271, 10)))
            data_ready = len(st.session_state.collected_malus_data) >= total_required
            btn_label = "📊 实验数据预览" if data_ready else "📊 请先采集完所有数据"
            if st.button(btn_label, use_container_width=True, disabled=not data_ready):
                show_malus_data(st.session_state.bg_noise)

    # 电脑端布局
    else:
        col_plot, col_control = st.columns([3, 1])
        with col_control:
            btn1, btn2 = st.columns(2)
            with btn1:
                if st.button("实验原理", use_container_width=True):
                    show_principle()
            with btn2:
                if st.button("使用说明", use_container_width=True):
                    show_guide()

            if st.button("重置", use_container_width=True):
                st.session_state.polarizer_angle = 0
                st.session_state.analyzer_angle = 90
                st.session_state.bg_noise = 0.1
                st.session_state.collected_malus_data = []
                st.session_state.reset_counter += 1
                st.rerun()

            m1, m2 = st.columns(2)
            with m1:
                t = "primary" if st.session_state.mode == "demo" else "secondary"
                if st.button("偏振光产生和检验", type=t, use_container_width=True):
                    st.session_state.mode = "demo"
                    st.rerun()
            with m2:
                t = "primary" if st.session_state.mode == "sim" else "secondary"
                if st.button("马吕斯定律", type=t, use_container_width=True):
                    st.session_state.mode = "sim"
                    st.session_state.analyzer_angle = 90
                    st.session_state.collected_malus_data = []
                    st.rerun()

            if st.session_state.mode == "demo":
                c1, c2 = st.columns([1, 2])
                with c1: st.markdown("**起偏器角度**")
                with c2:
                    st.session_state.polarizer_angle = st.number_input("起偏器", 0, 360,
                                                                       st.session_state.polarizer_angle, 10, "%d",
                                                                       label_visibility="collapsed",
                                                                       key=f"p{st.session_state.reset_counter}")

            if st.session_state.mode == "sim":
                c5, c6 = st.columns([1, 2])
                with c5: st.markdown("**环境光噪声**")
                with c6:
                    st.session_state.bg_noise = st.number_input("环境光噪声", min_value=0.10, max_value=0.30,
                                                                value=st.session_state.bg_noise, step=0.02,
                                                                format="%.2f", label_visibility="collapsed",
                                                                key=f"bg{st.session_state.reset_counter}")

            c3, c4 = st.columns([1, 2])
            with c3:
                st.markdown("**检偏器角度**")
            with c4:
                if st.session_state.mode == "demo":
                    st.session_state.analyzer_angle = st.number_input("检偏器", 0, 360, st.session_state.analyzer_angle,
                                                                      10, "%d", label_visibility="collapsed",
                                                                      key=f"a{st.session_state.reset_counter}")
                else:
                    st.session_state.analyzer_angle = st.number_input("检偏器_malus", 90, 270,
                                                                      st.session_state.analyzer_angle, 10, "%d",
                                                                      label_visibility="collapsed",
                                                                      key=f"am{st.session_state.reset_counter}")

            if st.session_state.mode == "demo":
                st.markdown("演示结果导出")
                remain = get_remaining_uses("demo_export")
                disabled = remain <= 0
                col_export, col_dl = st.columns(2)
                with col_export:
                    label = f"导出PNG（{remain}次）" if remain > 0 else "导出PNG（已用完）"
                    if st.button(label, use_container_width=True, disabled=disabled):
                        if check_usage_limit("demo_export"):
                            add_usage_record("demo_export")
                            fig = plot_polarization_experiment(st.session_state.polarizer_angle,
                                                               st.session_state.analyzer_angle, 0.0)
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                            buf.seek(0)
                            img = Image.open(buf).convert("RGB")
                            img = add_image_watermark(img, st.session_state.polarizer_angle,
                                                      st.session_state.analyzer_angle)
                            out_buf = io.BytesIO()
                            img.save(out_buf, format="PNG")
                            st.session_state.export_png = out_buf.getvalue()
                            plt.close(fig)
                            st.success("✅ 图片已添加水印并生成")
                            st.rerun()
                with col_dl:
                    if "export_png" in st.session_state:
                        st.download_button("⬇️ 下载PNG", st.session_state.export_png, "偏振光实验图.png", "image/png",
                                           use_container_width=True)

            if st.session_state.mode == "sim":
                fig_scat = plot_malus_scatter(st.session_state.analyzer_angle, st.session_state.bg_noise)
                st.pyplot(fig_scat, use_container_width=True)
                plt.close(fig_scat)

                total_required = len(list(range(90, 271, 10)))
                data_ready = len(st.session_state.collected_malus_data) >= total_required
                btn_label = "📊 实验数据预览" if data_ready else "📊 请先采集完所有数据"
                if st.button(btn_label, use_container_width=True, disabled=not data_ready):
                    show_malus_data(st.session_state.bg_noise)

        with col_plot:
            bg = st.session_state.bg_noise if st.session_state.mode == "sim" else 0.0
            fig_main = plot_polarization_experiment(st.session_state.polarizer_angle, st.session_state.analyzer_angle,
                                                    bg)
            st.pyplot(fig_main, use_container_width=True)
            plt.close(fig_main)


if __name__ == "__main__":
    main()