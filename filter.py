import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import os

def adjust_contrast_brightness(img, contrast=1.0, brightness=0):
    # コントラストと明るさを調整
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    # return = img * contrast + brightness

def split_image(img, tile_size, overlap):
    # 画像をタイルに分割
    # overlapはタイル間の重なりのピクセル数 (^ ^)
    tiles = []
    img_h, img_w = img.shape[:2]
    step = tile_size - overlap
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            tile = img[y:y+tile_size, x:x+tile_size]
            # img[y ~ y+tile_size, x ~ x+tile_size]
            tiles.append((tile, x, y))
    return tiles

def process_tile(model, tile, conf_threshold):
    # 推論
    # 出力：results[0]->boxes.xyxy, boxes.conf, boxes.cls
    results = model(tile, conf=conf_threshold)

    processed_results = []
    for result in results[0].boxes:
        if result.cls == 0:  # person class
            # CUDA tensor -> CPU -> numpy に変換してから数値化
            xyxy = result.xyxy[0].detach().cpu().numpy()
            x0, y0, x1, y1 = xyxy
            score = float(result.conf[0].detach().cpu().item())
            processed_results.append((int(x0), int(y0), int(x1), int(y1), score))

    return processed_results

def apply_soft_nms(boxes, scores, sigma=0.5, thresh=0.001, iou_thresh=0.3):
    """Applies Soft-NMS."""
    keep = []
    N = len(boxes)
    for i in range(N):
        max_pos = i
        max_score = scores[i]

        for j in range(i + 1, N):
            if scores[j] > max_score:
                max_score = scores[j]
                max_pos = j

        boxes[i], boxes[max_pos] = boxes[max_pos], boxes[i]
        scores[i], scores[max_pos] = scores[max_pos], scores[i]

        keep.append(i)

        for j in range(i + 1, N):
            iou = compute_iou(boxes[i], boxes[j])

            if iou > iou_thresh:
                scores[j] *= np.exp(-(iou * iou) / sigma)  # ガウス減衰式

    keep = [k for k in keep if scores[k] >= thresh]

    return keep

def compute_iou(box1, box2):
    """Computes the IoU of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

def process_image(model, img_path, conf_threshold, tile_size=640, overlap=320, iou_threshold=0.5):
    # 画像の読み込み
    img = cv2.imread(img_path)
    if img is None:  # エラーハンドリング
        print(f"Error: Could not read the image from {img_path}")
        return

    annotated_img, _ = process_frame(
        model,
        img,
        conf_threshold,
        tile_size=tile_size,
        overlap=overlap,
        iou_threshold=iou_threshold,
    )

    # 結果を直接表示
    cv2.imshow(f"Detection on {Path(img_path).name}", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_frame(model, img, conf_threshold, tile_size=640, overlap=320, iou_threshold=0.5):
    # 画像のコントラストと明るさを調整
    img = adjust_contrast_brightness(img, contrast=1.0, brightness=0)

    # 画像をタイルに分割
    tiles = split_image(img, tile_size, overlap)

    # 各タイルを処理
    all_results = []
    for tile, x_offset, y_offset in tiles:
        # タイルで推論
        results = process_tile(model, tile, conf_threshold)
        for x0, y0, x1, y1, score in results:
            all_results.append((x0 + x_offset, y0 + y_offset, x1 + x_offset, y1 + y_offset, score))

    # Soft NMSを適用して重複を除去
    if len(all_results) > 0:
        boxes = np.array([[r[0], r[1], r[2], r[3]] for r in all_results])
        scores = np.array([r[4] for r in all_results])
        keep = apply_soft_nms(boxes, scores, iou_thresh=iou_threshold)

        final_results = [all_results[i] for i in keep]
    else:
        final_results = []

    # 結果をオリジナル画像に描画
    for x0, y0, x1, y1, score in final_results:
        color = (0, 255, 0)
        text = f'person: {score:.1f}'
        img = cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        img = cv2.putText(img, text, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img, final_results

def main():
    # モデルのロード
    model = YOLO("yolo26n.pt")

    print('os.name =', os.name)
    print('cwd =', Path.cwd())

    # # Colab の場合は Drive をマウント
    # if 'COLAB_RELEASE_TAG' in os.environ:
    #     try:
    #         from google.colab import drive
    #         drive.mount('/content/drive', force_remount=False)
    #     except Exception as e:
    #         logger.warning(f'Drive mount skipped: {e}')

    # ここだけ変更すれば、処理対象を切り替えられます（単一画像 or フォルダ）
    img_source_path_str = '/content/drive/MyDrive/Colab Notebooks/DJ/YOLO/images/2026-02-06 225614.png'  # ここを変更
    img_source_path = Path(img_source_path_str)

    # if not img_source_path.exists():
    #     logger.error(f'Error: Path does not exist: {img_source_path_str}')
    #     logger.error(f'cwd: {Path.cwd()}')
    #     return

    # print('selected:', img_source_path)
    # print('is_file:', img_source_path.is_file(), 'is_dir:', img_source_path.is_dir())

    # 信頼度の閾値
    conf_threshold = 0.6

    # フォルダ内のすべての画像を処理、または単一画像を処理
    if img_source_path.is_file():
        process_image(model, str(img_source_path), conf_threshold)
    elif img_source_path.is_dir():
        for img_path in img_source_path.glob('*.png'):  # pngファイルを対象としていますが、他の拡張子も必要なら追加できます
            process_image(model, str(img_path), conf_threshold)
    else:
        print(f"Error: Invalid image source path: {img_source_path_str}. Must be a file or directory.")

if __name__ == '__main__':
    main()