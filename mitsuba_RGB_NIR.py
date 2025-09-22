import mitsuba as mi
import math
import os
import numpy as np

xml_file= "leather_chair"

def set_variant_and_load_scene(spectrum='rgb'):
    if spectrum == 'rgb':
        mi.set_variant("scalar_rgb")
        return mi.load_file(f"mitsuba_from_blender/{xml_file}.xml")
    elif spectrum == 'nir':
        mi.set_variant("scalar_spectral")  # NIR은 spectral 모드 필요
        return mi.load_file(f"mitsuba_from_blender/{xml_file}_nir.xml")

    
def make_camera_pose(scene, azim_deg, elev_deg, radius=4.4):
    # Bounding box 중심 계산
    bbox = scene.bbox()
    center = 0.5 * (bbox.min + bbox.max)
    target = mi.Point3f(center)

    # 각도 → 라디안 변환
    azim = math.radians(azim_deg)
    elev = math.radians(elev_deg)

    # 카메라 위치 계산 (center 기준으로 offset 적용)
    # x = radius * math.cos(elev) * math.cos(azim) + center.x
    # y = radius * math.cos(elev) * math.sin(azim) + center.y
    # z = radius * math.sin(elev) + center.z
    # 카메라 위치 계산 (center 기준으로 offset 적용), mitsuba 3D 좌표계는 Y축이 위쪽
    x = radius * math.cos(elev) * math.cos(azim) + center.x
    y = radius * math.sin(elev) + center.y
    z = radius * math.cos(elev) * math.sin(azim) + center.z

    origin = mi.Point3f([x, y, z])
    up = mi.Point3f([0.0, 1.0, 0.0])

    return mi.Transform4f().look_at(origin, target, up)


def render_view(spectrum, azim, elev, filename):
    scene = set_variant_and_load_scene(spectrum)
    camera_pose = make_camera_pose(scene, azim, elev, 15)
    # R = mi.Transform4f().rotate(mi.Point3f([1, 0, 0]), -90.0)
    if spectrum != 'nir':
        camera_pose = make_camera_pose(scene, azim, elev, 30)
        sensor_dict = {
            "type": "perspective",
            "to_world": camera_pose,
            "film": {
                "type": "hdrfilm",
                "width": 1024,
                "height": 1024,
                "rfilter": { "type": "gaussian" }
            },
            "sampler": {
                "type": "independent",
                "sample_count": 30
            }
        }
        sensor = mi.load_dict(sensor_dict)
        image = mi.render(scene, sensor=sensor)
    else:
        camera_pose = make_camera_pose(scene, azim, elev, 27)
        params = mi.traverse(scene)

        params['elm__3.to_world'] = camera_pose
        params["elm__3.film.size"] = [1024, 1024]
        params["elm__3.film.crop_size"] = [1024, 1024]
        params.update()
        sensor = scene.sensors()[0]
        image = mi.render(scene)


    os.makedirs("output_mat", exist_ok=True)
    output_path = os.path.join("output_mat", filename)

    if spectrum == 'rgb':
        mi.Bitmap(image).convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8
        ).write(output_path)
    else:
        # IR은 단일 파장 렌더링 결과로 (H, W) float32 배열임
        mi.Bitmap(image).convert(
        pixel_format=mi.Bitmap.PixelFormat.Y,
        component_format=mi.Struct.Type.UInt8
    ).write(output_path)

    print(f"Saved: {output_path}")

    # 월드 좌표계에서의 픽셀 좌표 계산
# ① 필름 해상도 가져오기
    width, height = sensor.film().crop_size()

    # ② 월드 좌표 저장 배열 초기화 (NaN으로 채움)
    coords = np.full((height, width, 3), np.nan, dtype=np.float32)

    # ③ 각 픽셀 순회
    for y in range(height):
        for x in range(width):
            # 픽셀 좌표를 [0,1] 정규화 → Point2f 변환
            pixel_sample = mi.Point2f((x + 0.5) / width, (y + 0.5) / height)

            # 렌즈 샘플 좌표 (고정값 사용)
            lens_sample = mi.Point2f(0.5, 0.5)

            # 레이 생성 (time=0.0, wavelength_sample=0.0)
            ray, _ = sensor.sample_ray(
                0.0,              # time
                0.0,              # wavelength_sample (0~1)
                pixel_sample,     # 픽셀 위치
                lens_sample       # 렌즈 위치
            )

            # 레이와 씬의 교차점 계산
            si = scene.ray_intersect(ray)

            # 유효한 교차점이면 저장
            if si.is_valid():
                coords[y, x] = [si.p.x, si.p.y, si.p.z] # TODO: Maybe [si.p.x, si.p.z, si.p.y] is correct?
    return coords

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def show_rgb_nir_comparison(azim=60, elev=45, rgb_folder="output_mat", nir_folder="output_mat"):
    rgb_path = os.path.join(rgb_folder, f'render_rgb_azim{azim}_elev{elev}.png')
    nir_path = os.path.join(nir_folder, f'render_ir850nm_azim{azim}_elev{elev}.png')

    # 이미지 로드
    rgb_img = mpimg.imread(rgb_path)
    nir_img = mpimg.imread(nir_path)

    # Figure 생성, 1행 2열
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # RGB 이미지 출력
    axs[0].imshow(rgb_img)
    axs[0].axis('off')
    axs[0].set_title('RGB Image', fontsize=16)

    # NIR 이미지 출력 (흑백이므로 cmap='gray')
    axs[1].imshow(nir_img, cmap='gray')
    axs[1].axis('off')
    axs[1].set_title('NIR Image (850nm)', fontsize=16)

    plt.tight_layout()
    plt.show()

# 사용 예시

def visualize_point_cloud(points):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 샘플링 (너무 많으면 느리니까)
    max_points = 200_000
    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=0.5, c=points[:, 2], cmap='viridis', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Multi-view Point Cloud from Mitsuba Renders")
    plt.show()


import shutil
if __name__ == "__main__":
    # azimuths = [0, 60, 120, 180, 240, 300]
    # elevations = [45, 0, -45]
    # azimuths = [0, 90, 180, 270]
    # elevations = [0, 45]
    azimuths = [ -40, -80, -130]
    elevations = [0, 45]
    output_np_path = "output_np"
    shutil.rmtree(output_np_path, ignore_errors=True)
    os.makedirs(output_np_path, exist_ok=True)
    points = []
    for elev in elevations:
        for azim in azimuths:
            coords = render_view('rgb', azim, elev, f'render_rgb_azim{azim}_elev{elev}.png')
            render_view('nir', azim, elev, f'render_nir_azim{azim}_elev{elev}.png')
            np.save(os.path.join(output_np_path, f'azim{azim}_elev{elev}.npy'), coords)
    # show_rgb_nir_comparison(azim=90, elev=45)
    for npy_file in sorted(os.listdir(output_np_path)):
        if npy_file.endswith('.npy'):
            coords = np.load(os.path.join(output_np_path, npy_file))
            points.append(coords)
    points_np = np.stack(points, axis=0)
    print("shape: ", points_np.shape)  # (num_views, height, width, 3)
    np.save("/home/han/workspace/VLM-LLM/projected_views_pcd/points.npy", points_np)
    points = points_np.reshape(-1, 3)
    mask = ~np.isnan(points[:, 0])               # x좌표 NaN 여부로 필터링
    points = points[mask]
    print(f"Total points: {points.shape[0]}")

    visualize_point_cloud(points)