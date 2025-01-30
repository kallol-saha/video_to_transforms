import ffmpeg
import sys
import argparse

def slow_video(input_file, output_file, slow_factor):
    try:
        # Input video stream
        input_stream = ffmpeg.input(input_file)
        
        # Apply interpolation and slow down the video
        # Get the fps of the input video
        probe = ffmpeg.probe(input_file)
        video_stream_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        input_fps = eval(video_stream_info['r_frame_rate'])

        output_stream = (
            input_stream
            .filter('minterpolate', mi_mode='mci', fps=input_fps * slow_factor)
            .output(output_file)
        )
        
        # Run ffmpeg
        ffmpeg.run(output_stream)
        print(f"Video has been slowed down by a factor of {slow_factor} and saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Slow down a video using interpolation.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input video file (mkv or mp4)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output video file')
    parser.add_argument('--slow_factor', type=float, default=2.0, help='Factor by which to slow down the video')

    args = parser.parse_args()

    if not args.input_file.endswith(('.mkv', '.mp4')):
        print("Input file must be an mkv or mp4 file")
        sys.exit(1)
    
    slow_video(args.input_file, args.output_file, args.slow_factor)