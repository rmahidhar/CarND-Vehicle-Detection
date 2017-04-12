from classifier import Classifier
from moviepy.editor import VideoFileClip
from vehicle_detector import VehicleDetector
import matplotlib.image
import time
import glob

start = time.time()
classifier = Classifier()
latency = time.time() - start
print("training time time:{0:.2f}".format(latency))
images = glob.glob('test_images/*.jpg')
for image_name in images:
    image = matplotlib.image.imread(image_name)
    detector = VehicleDetector(image, classifier)
    start = time.time()
    image = detector.detect_vehicles(image)
    latency = time.time() - start
    print("image_name process time:{0:.2f}".format(latency))
    #ImagePlotter()(image, title=image_name)

video_output_name = 'project_video_annotated_vehicles.mp4'
video = VideoFileClip("project_video.mp4")
#video = VideoFileClip("project_video.mp4").subclip(38,43)
tracker = VehicleDetector(video.get_frame(0), classifier)
video_output = video.fl_image(tracker.detect_vehicles)
video_output.write_videofile(video_output_name, audio=False)