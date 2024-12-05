import gradio as gr
from pathlib import Path
import time
from datetime import datetime
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from loguru import logger
import threading
import http.server
import socketserver
import os

# Hàm sinh video
def generate_video(
    video_size, video_length, infer_steps, prompt, flow_reverse, seed, use_cpu_offload, embedded_guidance_scale, save_path
):
    try:
        # Xác thực các tham số đầu vào
        width, height = map(int, video_size.split(" "))  # Tách và xác thực kích thước video
        if width <= 0 or height <= 0:
            return "Lỗi: Chiều rộng và chiều cao phải là số nguyên dương."

        video_length = int(video_length)
        if (video_length - 1) % 4 != 0:  # Xác thực video_length
            return f"Lỗi: `(video_length - 1)` phải chia hết cho 4, hiện tại nhận giá trị {video_length}"

        infer_steps = int(infer_steps)
        seed = int(seed)
        flow_reverse = bool(flow_reverse)
        use_cpu_offload = bool(use_cpu_offload)
        embedded_guidance_scale = float(embedded_guidance_scale) if embedded_guidance_scale is not None else 1.0  # Mặc định là 1.0
        save_path = save_path or "./results"

        # Giá trị cố định cho guidance scale
        guidance_scale = 1.0  # Cố định là 1

        # Xây dựng các tham số mặc định
        args = parse_args()
        args.model = "HYVideo-T/2-cfgdistill"
        args.video_size = (width, height)
        args.video_length = video_length
        args.infer_steps = infer_steps
        args.prompt = prompt
        args.flow_reverse = flow_reverse
        args.seed = seed
        args.use_cpu_offload = use_cpu_offload
        args.embedded_guidance_scale = embedded_guidance_scale  # Đảm bảo giá trị hợp lệ
        args.cfg_scale = guidance_scale  # Cố định là 1
        args.save_path = Path(save_path)

        if not args.save_path.exists():
            args.save_path.mkdir(parents=True, exist_ok=True)

        # Tải mô hình
        models_root_path = Path("ckpts")
        hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)

        # Bắt đầu tạo video
        outputs = hunyuan_video_sampler.predict(
            prompt=args.prompt,
            height=args.video_size[1],
            width=args.video_size[0],
            video_length=args.video_length,
            seed=args.seed,
            infer_steps=args.infer_steps,
            flow_reverse=args.flow_reverse,
            guidance_scale=args.cfg_scale,  # Cố định là 1
            embedded_guidance_scale=args.embedded_guidance_scale,
        )
        samples = outputs["samples"]

        # Lưu video
        video_files = []
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_file = args.save_path / f"{time_flag}_seed{outputs['seeds'][i]}.mp4"
            save_videos_grid(sample, str(save_file), fps=24)
            logger.info(f"Video đã được lưu tại: {save_file}")
            video_files.append(save_file)

        return video_files

    except Exception as e:
        logger.error(f"Lỗi: {e}")
        return str(e)

# Giao diện Gradio
interface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Textbox(label="Kích thước Video (ví dụ: 540 960)", value="540 960"),  # --video-size
        gr.Slider(
            5, 257, step=4, label="Độ dài Video", value=129
        ),  # --video-length; chỉ cho phép bội số của 4 (+1)
        gr.Slider(1, 100, step=1, label="Số bước suy luận", value=30),  # --infer-steps
        gr.Textbox(label="Nội dung Prompt", value="Một cô gái trẻ đang đi dưới mưa..."),  # --prompt
        gr.Checkbox(label="Flow Reverse", value=True),  # --flow-reverse
        gr.Number(label="Seed", value=0),  # --seed
        gr.Checkbox(label="Dùng CPU Offload", value=True),  # --use-cpu-offload
        gr.Slider(0.0, 10.0, step=0.1, label="Embedded Guidance Scale", value=1.0),  # --embedded-guidance-scale
        gr.Textbox(label="Đường dẫn lưu file", value="./results"),  # --save-path
    ],
    outputs=[
        gr.File(label="Video đã tạo"),  # Đầu ra video
    ],
    title="Hunyuan Video Generator",
    description="Tạo video với fixed guidance scale là 1 và độ dài video được xác thực.",
)

if __name__ == "__main__":
    # Tuỳ chọn: Khởi chạy server cục bộ để lưu trữ video
    def start_server():
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("0.0.0.0", 8000), handler) as httpd:
            print("Server đang chạy tại cổng 8000")
            httpd.serve_forever()

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Khởi chạy giao diện Gradio
    interface.launch(share=True)
