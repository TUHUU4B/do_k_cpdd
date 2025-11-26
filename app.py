import math
from dataclasses import dataclass

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Độ chặt cấp phối đá dăm - AASHTO T191",
    page_icon="🧱",
    layout="wide",
)


@dataclass
class SandCalibration:
    """Represents calibration results for the sand-cone method."""

    density: float  # g/cm3
    mass_sand: float | None = None
    volume_container: float | None = None


@st.cache_data
def format_number(value: float, digits: int = 2) -> str:
    """Small helper to keep number formatting consistent (Vietnamese format)."""
    if math.isnan(value):
        return "N/A"
    # Format with Vietnamese style: dot for thousands, comma for decimal
    formatted = f"{value:,.{digits}f}"
    # Replace comma with dot for thousands separator, then replace dot with comma for decimal
    parts = formatted.split(".")
    if len(parts) == 2:
        integer_part = parts[0].replace(",", ".")
        decimal_part = parts[1]
        return f"{integer_part},{decimal_part}"
    else:
        # No decimal part
        return parts[0].replace(",", ".")


def compute_sand_density(
    mode: str,
    known_density: float,
    mass_full: float,
    mass_empty: float,
    calibration_volume: float,
) -> SandCalibration:
    """Compute sand density based on the user-selected mode."""
    if mode == "Nhập trực tiếp":
        return SandCalibration(density=known_density)

    mass_sand = mass_full - mass_empty
    if calibration_volume <= 0 or mass_sand <= 0:
        return SandCalibration(density=float("nan"))

    density = mass_sand / calibration_volume
    return SandCalibration(
        density=density,
        mass_sand=mass_sand,
        volume_container=calibration_volume,
    )


def compute_moisture_content(
    mode: str,
    known_moisture: float,
    mass_wet_sample: float,
    mass_dry_sample: float,
) -> float:
    """Compute moisture content based on the user-selected mode."""
    if mode == "Nhập trực tiếp":
        return known_moisture

    if mass_dry_sample <= 0:
        return float("nan")

    moisture_percent = (mass_wet_sample - mass_dry_sample) / mass_dry_sample * 100.0
    return moisture_percent


def compute_field_results(
    rho_sand: float,
    mass_before: float,
    mass_after: float,
    mass_base_plate: float,
    mass_wet_soil: float,
    moisture_percent: float,
) -> dict[str, float]:
    """Core AASHTO T191 calculations."""
    mass_sand_hole = mass_before - mass_after - mass_base_plate
    volume_hole = mass_sand_hole / rho_sand if rho_sand > 0 else float("nan")
    gamma_wet = mass_wet_soil / volume_hole if volume_hole > 0 else float("nan")

    moisture_ratio = moisture_percent / 100.0
    gamma_dry = gamma_wet / (1 + moisture_ratio) if gamma_wet > 0 else float("nan")

    return {
        "mass_sand_hole": mass_sand_hole,
        "volume_hole": volume_hole,
        "gamma_wet": gamma_wet,
        "gamma_dry": gamma_dry,
    }


def main() -> None:
    st.title("Tính độ chặt cấp phối đá dăm (AASHTO T191)")
    st.caption(
        "Ứng dụng hỗ trợ hiện trường xác định khối lượng thể tích khô của cấp phối "
        "đá dăm theo phương pháp rót cát."
    )

    with st.sidebar:
        # Logo và thông tin công ty
        try:
            st.image("logo.png", use_container_width=True)
        except FileNotFoundError:
            st.warning("Không tìm thấy file logo.png")
        
        st.markdown(
            "<div style='text-align: center; margin-top: 10px; margin-bottom: 10px;'>"
            "<h4>CÔNG TY TỨ HỮU</h4>"
            "<p style='font-size: 0.9em; color: #666;'>Tác giả: MR Tuấn - 0946135156</p>"
            "</div>",
            unsafe_allow_html=True
        )
        st.divider()
        
        st.header("Hướng dẫn nhanh")
        st.markdown(
            "- Chuẩn hóa cát rót trước khi ra hiện trường.\n"
            "- Ghi lại khối lượng từng bước theo gam.\n"
            "- Xác định độ ẩm từ mẫu đại diện (sấy khô ở 105-110°C).\n"
            "- Hoặc nhập trực tiếp độ ẩm từ phòng thí nghiệm."
        )

    st.subheader("Khối lượng thể tích khô lớn nhất (γdmax)")
    target_gamma = st.number_input(
        "Khối lượng thể tích khô lớn nhất (Proctor Test Number) (g/cm³)",
        min_value=0.0,
        value=2.354,
        step=0.001,
        format="%.3f",
    )
    if target_gamma > 0:
        st.info(f"Giá trị đã nhập: **{format_number(target_gamma, 3)}** g/cm³")

    st.divider()
    sand_mode = st.radio(
            "Chọn cách xác định khối lượng riêng của cát chuẩn",
            ("Nhập trực tiếp", "Tính từ thí nghiệm chuẩn"),
            horizontal=True,
        )

    col1, col2, col3 = st.columns(3)
    if sand_mode == "Nhập trực tiếp":
        known_density = col1.number_input(
            "Khối lượng riêng cát chuẩn ρsand (g/cm³)",
            min_value=0.0,
            value=1.58,
            step=0.001,
            format="%.3f",
        )
        calibration = compute_sand_density(
            sand_mode, known_density, 0.0, 0.0, 1.0
        )
    else:
        mass_full = col1.number_input(
            "Khối lượng bình + cát (g)", min_value=0.0, value=5304.0, step=1.0
        )
        mass_empty = col2.number_input(
            "Khối lượng bình rỗng (g)", min_value=0.0, value=2300.0, step=1.0
        )
        calibration_volume = col3.number_input(
            "Thể tích bình chuẩn (cm³)",
            min_value=0.0,
            value=2000.0,
            step=1.0,
        )
        calibration = compute_sand_density(
            sand_mode,
            0.0,
            mass_full,
            mass_empty,
            calibration_volume,
        )

    if math.isnan(calibration.density) or calibration.density <= 0:
        st.error("Vui lòng kiểm tra lại dữ liệu hiệu chuẩn cát.")
        rho_sand = 0.0
    else:
        rho_sand = calibration.density
        st.success(f"ρsand = {format_number(rho_sand, 3)} g/cm³")

    st.divider()
    st.subheader("2. Xác định độ ẩm")
    moisture_mode = st.radio(
        "Chọn cách xác định độ ẩm",
        ("Nhập trực tiếp", "Tính từ mẫu thí nghiệm"),
        horizontal=True,
        key="moisture_mode",
    )

    col_m1, col_m2 = st.columns(2)
    if moisture_mode == "Nhập trực tiếp":
        known_moisture = col_m1.number_input(
            "Độ ẩm w (%)",
            min_value=0.0,
            value=4.5,
            step=0.1,
            key="known_moisture",
        )
        moisture_percent = compute_moisture_content(
            moisture_mode, known_moisture, 0.0, 1.0
        )
    else:
        mass_wet_sample = col_m1.number_input(
            "Khối lượng mẫu ẩm (g)",
            min_value=0.0,
            value=200.0,
            step=0.1,
            key="mass_wet_sample",
        )
        mass_dry_sample = col_m2.number_input(
            "Khối lượng mẫu khô (g)",
            min_value=0.0,
            value=191.0,
            step=0.1,
            key="mass_dry_sample",
        )
        moisture_percent = compute_moisture_content(
            moisture_mode, 0.0, mass_wet_sample, mass_dry_sample
        )

    if math.isnan(moisture_percent) or moisture_percent < 0:
        st.error("Vui lòng kiểm tra lại dữ liệu xác định độ ẩm.")
        moisture_percent = 0.0
    else:
        st.success(f"Độ ẩm w = {format_number(moisture_percent, 2)} %")

    st.divider()
    st.subheader("3. Dữ liệu hiện trường")
    col_a, col_b, col_c = st.columns(3)
    mass_before = col_a.number_input(
        "Khối lượng bộ dụng cụ + cát trước thí nghiệm (g)",
        min_value=0.0,
        value=8000.0,
        step=1.0,
    )
    mass_after = col_b.number_input(
        "Khối lượng bộ dụng cụ + cát sau thí nghiệm (g)",
        min_value=0.0,
        value=5100.0,
        step=1.0,
    )
    mass_base_plate = col_c.number_input(
        "Khối lượng cát trong phểu rót (g)",
        min_value=0.0,
        value=1400.0,
        step=1.0,
    )

    mass_wet_soil = st.number_input(
        "Khối lượng mẫu đất/đá ẩm lấy từ hố (g)",
        min_value=0.0,
        value=2500.0,
        step=1.0,
    )

    results = compute_field_results(
        rho_sand,
        mass_before,
        mass_after,
        mass_base_plate,
        mass_wet_soil,
        moisture_percent,
    )

    if results["mass_sand_hole"] <= 0:
        st.warning("Khối lượng cát vào hố ≤ 0. Kiểm tra lại số liệu cân.")
    elif results["volume_hole"] <= 0:
        st.warning("Thể tích hố không hợp lệ. Kiểm tra ρsand.")
    else:
        st.success("Đã tính toán xong. Xem bảng kết quả bên dưới.")

    data = {
        "Thông số": [
            "Độ ẩm w (%)",
            "Khối lượng cát trong hố (g)",
            "Thể tích hố (cm³)",
            "Khối lượng thể tích ẩm γ (g/cm³)",
            "Khối lượng thể tích khô γd (g/cm³)",
        ],
        "Giá trị": [
            format_number(moisture_percent, 2),
            format_number(results["mass_sand_hole"]),
            format_number(results["volume_hole"]),
            format_number(results["gamma_wet"], 3),
            format_number(results["gamma_dry"], 3),
        ],
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if target_gamma > 0 and results["gamma_dry"] > 0:
        compaction_percent = results["gamma_dry"] / target_gamma * 100
        st.metric(
            "Độ chặt so với yêu cầu (%)",
            format_number(compaction_percent, 1),
        )
    elif target_gamma > 0:
        st.info("Chưa có đủ dữ liệu để so sánh với γd yêu cầu.")


if __name__ == "__main__":
    main()

