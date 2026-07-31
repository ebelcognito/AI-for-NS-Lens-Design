import base64
import sys
from pathlib import Path

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_rl.utils.ray_optics_bridge import run_simulation
from optical_rl.utils.scene_builder import scene_angled_ray_with_quad_at_y


def _save_first_image(result, output_path: Path):
    images = result.get("images", [])
    assert images, "No images returned from ray-optics runner."
    data_url = images[0].get("dataUrl")
    assert data_url, "Image missing dataUrl."
    png_data = data_url.split(",")[1]
    png_bytes = base64.b64decode(png_data)
    output_path.write_bytes(png_bytes)


def test_angled_ray_vertical_quad_sweep():
    output_dir = Path("results") / "angled_ray_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    quad_centers_y = [0, 20, 40, 60]
    for quad_y in quad_centers_y:
        scene = scene_angled_ray_with_quad_at_y(quad_y)
        result, _ = run_simulation(scene)
        power = result["detectors"][0]["power"]
        assert 0 <= power <= 1
        output_path = output_dir / f"quad_y{quad_y}.png"
        _save_first_image(result, output_path)
