import io
import openai
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, send_file
import uuid
import base64
from threading import Thread
import logging

from StatusCodes import StatusCodes
from Service import call_openai_vision, generate_music
from Transactions import transactions
from Messages import Messages

app = Flask(__name__)

apiVersion = "v1"
contextRoot = f"/api/{apiVersion}"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

last_transaction_id = None


@app.route(f'{contextRoot}/healthcheck', methods=['GET'])
def health_check():
    return jsonify({"status": "up"}), 200


@app.route(f'{contextRoot}/upload', methods=['POST'])
def upload_image():
    return handle_uploaded_image(False)


@app.route(f'{contextRoot}/upload-dall-e', methods=['POST'])
def upload_image_dall_e():
    return handle_uploaded_image(True)


def handle_uploaded_image(use_dall_e):
    global last_transaction_id
    logging.info('Received upload request')
    if 'inputFile' not in request.files:
        logging.debug('Cancelling upload request because of no input file sent')
        return jsonify({"message": Messages.NO_INPUT_FILE_SENT}), 400

    input_file = request.files['inputFile']

    if input_file.filename == '':
        logging.debug('Cancelling upload request because of no input file found')
        return jsonify({"message": Messages.INPUT_FILE_NOT_FOUND}), 404

    # Convert image to PIL
    image = Image.open(input_file.stream)

    image_format = image.format
    if not image_format:
        logging.debug('Cancelling upload request because no format could be determined')
        return jsonify({"message": Messages.UNABLE_DETERMINING_IMAGE_FORMAT}), 400

    image_format = image_format.lower()
    if image_format not in ['png', 'jpeg', 'gif', 'webp']:
        logging.warning('Image format was manually casted to JPEG')
        image_format = 'jpeg'
    else:
        image_format = image.format

    # Convert PI to Bytes object
    buffered = BytesIO()
    image.save(buffered, format=image_format)
    img_byte = buffered.getvalue()

    if len(img_byte) > 20 * 1024 * 1024:
        logging.debug('Cancelling upload request because image file exceeds 20MB')
        return jsonify({"message": Messages.FILE_SIZE_EXCEEDING}), 413

    # Convert Bytes object to image encoded with Base64
    img_base64 = base64.b64encode(img_byte).decode('utf-8')

    transaction_id = str(uuid.uuid4())
    last_transaction_id = transaction_id
    transactions[transaction_id] = {
        "status": StatusCodes.RECEIVED.value,
        "image": img_base64,
        "analysis": None,
        "keywords": [],
        "score": None,
        "error": None,
        "generatedMusic": None,
        "musicStatus": StatusCodes.IDLING.value
    }

    # Start async task
    thread = Thread(target=async_call_openai_vision, args=(transaction_id, use_dall_e))
    thread.start()

    return jsonify({"message": Messages.IMAGE_RECEIVED, "transaction_id": transaction_id}), 200


# Wrapper function for asynchronous execution
def async_call_openai_vision(transaction_id, fast_processing):
    try:
        call_openai_vision(transaction_id, fast_processing)
        logging.info('✅ Generation process done')
    except openai.BadRequestError as e:
        logging.error(f'Caught BadRequestError while calling OpenAI Vision: {e}')
        transactions[transaction_id]["status"] = StatusCodes.ERROR.value
        transactions[transaction_id]["error"] = Messages.OPENAI_DENIAL
    except Exception as e:
        logging.error(f'Caught error while calling OpenAI Vision: {e}')
        transactions[transaction_id]["status"] = StatusCodes.ERROR.value
        transactions[transaction_id]["error"] = Messages.ERROR_INPUT_FILE


@app.route(f'{contextRoot}/analysis/<transaction_id>', methods=['GET'])
def get_analysis(transaction_id):
    logging.info('Received analysis request')
    if transaction_id not in transactions:
        logging.debug(f'Cancelling analysis request because no transaction with id {transaction_id} was found')
        return jsonify({"error": Messages.TRANSACTION_NOT_FOUND_OR_INVALID}), 404

    status = transactions[transaction_id]["status"]
    if status == StatusCodes.ERROR.value:
        logging.debug('Cancelling analysis request because transaction is in error')
        return jsonify({"message": Messages.TRANSACTION_IN_ERROR}), 409
    if status == StatusCodes.RUNNING_ANALYSIS.value or status == StatusCodes.RECEIVED.value:
        logging.debug('Cancelling analysis request because transaction has no content')
        return jsonify({"message": Messages.NO_CONTENT}), 204

    return jsonify({"transaction_id": transaction_id, "analysis": transactions[transaction_id]["keywords"]}), 200


@app.route(f'{contextRoot}/analysis-intern/<transaction_id>', methods=['GET'])
def get_analysis_intern(transaction_id):
    logging.info('Received analysis-intern request')
    if transaction_id not in transactions:
        logging.debug(f'Cancelling analysis-intern request because no transaction with id {transaction_id} was found')
        return jsonify({"error": Messages.TRANSACTION_NOT_FOUND_OR_INVALID}), 404

    status = transactions[transaction_id]["status"]
    if status == StatusCodes.ERROR.value:
        logging.debug('Cancelling analysis-intern request because transaction is in error')
        return jsonify({"message": Messages.TRANSACTION_IN_ERROR}), 409
    if status == StatusCodes.RUNNING_ANALYSIS.value or status == StatusCodes.RECEIVED.value:
        logging.debug('Cancelling analysis-intern request because transaction has no content')
        return jsonify({"message": Messages.NO_CONTENT}), 204

    return jsonify({"transaction_id": transaction_id, "analysis": transactions[transaction_id]["analysis"]}), 200


@app.route(f'{contextRoot}/score/<transaction_id>', methods=['GET'])
def get_score(transaction_id):
    logging.info('Received score request')
    if transaction_id not in transactions:
        logging.debug(f'Cancelling score request because no transaction with id {transaction_id} was found')
        return jsonify({"error": Messages.TRANSACTION_NOT_FOUND_OR_INVALID}), 404

    status = transactions[transaction_id]["status"]
    if status == StatusCodes.ERROR.value:
        logging.debug('Cancelling score request because transaction is in error')
        return jsonify({"message": Messages.TRANSACTION_IN_ERROR}), 409
    if (status == StatusCodes.RUNNING_ANALYSIS.value or
            status == StatusCodes.RUNNING_GENERATION.value or
            status == StatusCodes.RECEIVED.value or
            status == StatusCodes.IDLING.value):
        logging.debug('Cancelling score request because transaction has no content')
        return jsonify({"message": Messages.NO_CONTENT}), 204

    base64_encoded_image = transactions[transaction_id]["score"]
    image_data = base64.b64decode(base64_encoded_image)
    img_byte_arr = io.BytesIO(image_data)

    return send_file(
        img_byte_arr,
        mimetype='image/png',
        as_attachment=True,
        download_name='score.png'
    ), 200


@app.route(f'{contextRoot}/last-score', methods=['GET'])
def get_last_request():
    if last_transaction_id is None or last_transaction_id not in transactions:
        return jsonify({"message": Messages.NO_LAST_TRANSACTION_FOUND}), 404

    return get_score(last_transaction_id)


@app.route(f'{contextRoot}/status/<transaction_id>', methods=['GET'])
def get_status(transaction_id):
    logging.info('Received status request')
    if transaction_id not in transactions:
        logging.debug(f'Cancelling status request because no transaction with id {transaction_id} was found')
        return jsonify({"error": Messages.TRANSACTION_NOT_FOUND_OR_INVALID}), 404

    return jsonify({"transaction_id": transaction_id, "status": transactions[transaction_id]["status"]}), 200


@app.route(f'{contextRoot}/error/<transaction_id>', methods=['GET'])
def get_error(transaction_id):
    logging.info('Received error request')
    if transaction_id not in transactions:
        logging.debug(f'Cancelling error request because no transaction with id {transaction_id} was found')
        return jsonify({"error": Messages.TRANSACTION_NOT_FOUND_OR_INVALID}), 404

    status = transactions[transaction_id]["status"]
    if status != StatusCodes.ERROR:
        logging.debug('Cancelling error request because transaction is not in error')
        return jsonify({"error": Messages.NO_ERRORS}), 409

    return jsonify({"transaction_id": transaction_id, "error": transactions[transaction_id]["error"]}), 200


@app.route(f'{contextRoot}/music/<transaction_id>', methods=['GET'])
def music(transaction_id):
    logging.info('Received music request')
    if transaction_id not in transactions:
        logging.debug(f'Cancelling music request because no transaction with id {transaction_id} was found')
        return jsonify({"error": Messages.TRANSACTION_NOT_FOUND_OR_INVALID}), 404

    status = transactions[transaction_id]["musicStatus"]
    if status == StatusCodes.ERROR.value:
        logging.debug('Cancelling music request because transaction is in error')
        return jsonify({"message": Messages.TRANSACTION_IN_ERROR}), 409
    if (status == StatusCodes.RUNNING_GENERATION.value or
            status == StatusCodes.IDLING.value):
        logging.debug('Cancelling music request because transaction has no content')
        return jsonify({"message": Messages.NO_CONTENT}), 204

    audio_bytes = transactions[transaction_id]["generatedMusic"]

    return_data = io.BytesIO(audio_bytes)
    return_data.seek(0)

    return send_file(
        return_data,
        mimetype='audio/wav',
        as_attachment=True,
        download_name='music.wav'
    ), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4242)
