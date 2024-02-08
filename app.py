from flask import Flask, render_template, request, send_file, make_response
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64


app = Flask(__name__)


def process_image(image):
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Y_channel, Cr, Cb = cv2.split(ycrcb_image)
    Y_channel = clahe.apply(Y_channel)
    merged_ycrcb = cv2.merge([Y_channel, Cr, Cb])
    final_image = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)
    rgb_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    return rgb_image

def convert_image_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def flip_image(image):
    flipped_image = cv2.flip(image, 0)
    rgb_flipped_image = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)
    return rgb_flipped_image

def plot_histograms(original_image, processed_image):
    Y_original, Cr_original, Cb_original = cv2.split(cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb))
    Y_processed, Cr_processed, Cb_processed = cv2.split(cv2.cvtColor(processed_image, cv2.COLOR_BGR2YCrCb))
    channels = ('Y', 'Cr', 'Cb')
    fig, axs = plt.subplots(2, 3, figsize=(16, 6))
    for i, channel in enumerate([Y_original, Cr_original, Cb_original]):
        axs[0, i].hist(channel.ravel(), bins=256, range=[0, 256])
        axs[0, i].set_title(f'Original {channels[i]}')
    for i, channel in enumerate([Y_processed, Cr_processed, Cb_processed]):
        axs[1, i].hist(channel.ravel(), bins=256, range=[0, 256])
        axs[1, i].set_title(f'Processed {channels[i]}')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_stream = BytesIO(uploaded_file.read())
            image_stream.seek(0)
            file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            option = request.form.get('option')

            if option == 'Histogram Equalization':
                processed_image = process_image(image)
                histogram_image_buf = plot_histograms(image, processed_image)
                _, img_encoded = cv2.imencode('.png', processed_image)
                processed_img_base64 = base64.b64encode(img_encoded).decode('utf-8')
                histogram_img_base64 = base64.b64encode(histogram_image_buf.getvalue()).decode('utf-8')
                return render_template('results.html', processed_image=processed_img_base64, histogram_image=histogram_img_base64)

            elif option == '흑백변환':
                grayscale_image = convert_image_to_grayscale(image)
                _, img_encoded = cv2.imencode('.png', grayscale_image)
                grayscale_img_base64 = base64.b64encode(img_encoded).decode('utf-8')
                return render_template('results.html', processed_image=grayscale_img_base64, histogram_image='')

            elif option == '90도 회전':
                flipped_image = flip_image(image)
                _, img_encoded = cv2.imencode('.png', flipped_image)
                flipped_img_base64 = base64.b64encode(img_encoded).decode('utf-8')
                return render_template('results.html', processed_image=flipped_img_base64, histogram_image='')

    return render_template('index.html')

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)