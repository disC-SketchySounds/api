import base64
import configparser
from io import BytesIO

import requests
from torchvision.transforms import ToTensor

from PIL import Image
from openai import OpenAI
from StatusCodes import StatusCodes
from Transactions import transactions
from Messages import Messages
import logging
from diffusers import StableDiffusionImg2ImgPipeline
from compel import Compel
import torch
import io


def get_api_key():
    """
    Reads the API key from the config file
    """
    config = configparser.ConfigParser()
    with open("secure/openAI.properties") as stream:
        config.read_string("[top]\n" + stream.read())
    return config.get('top', 'openai.key')


api_key = get_api_key()
open_ai_client = OpenAI(api_key=api_key)

sd_model = "/nfs/scratch/students/kremlingph95027/content/v1-5-pruned-emaonly.safetensors"
lora_folder = "/nfs/scratch/students/kremlingph95027/stable-diffusion-webui/models/Lora"
lora_weights_1_name = "v15Tune_Lora_17800.safetensors"
lora_weights_2_name = "15Tune_Lora_17100.safetensors"
lora_weights_3_name = "v15Tune_Lora_12800.safetensors"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def call_openai_vision(transaction_id, use_dall_e):
    """
    Calls OpenAI Vision to run an image analysis

    Args:
        :param transaction_id: The ID of the transaction for which the analysis should be executed
        :param use_dall_e: Bool flag indicating whether SD (False) or DALL-E (True) should be used
    """
    transactions[transaction_id]["status"] = StatusCodes.RUNNING_ANALYSIS.value
    image = transactions[transaction_id]["image"]

    logging.info('Calling OpenAI Vision')

    response = open_ai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Als musikalische Bilderkennung
    analysiere dieses Bild auf Basis seiner visuellen Eigenschaften, die für die Generierung
    einer musikalischen Partitur relevant sein könnten
    wobei du die gefundenen Eigenschaften in einer komma-getrennten Liste herausschreibst.
    Achte auf verschiedene Farben und interpretiere diese als eigene Musiker.
    Achte besonders auf folgende Eigenschaften und baue sie falls passend ein: abstrakt, chaotisch, duester, freundlich,
    geschwungen, gleichmaessig, hoch, minimalistisch, tief. Beschränke dich jedoch nicht auf diese Eigenschaften.
    Gib das Ergebnis in folgendem Format aus: 'Farbe Form (Position): Eigenschaften'. 
    #Beispiel: 'Gelbe Linie (oben rechts): hoch'"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    logging.debug('OpenAI Vision Request successful')
    """
    Schwarzer Hintergrund
Grüne Linien (Mitte): chaotisch, geschwungen, freundlich
Blaue Linie/Gekritzel (unten Mitte): chaotisch, abstrakt
Lila Streifen (mittig): gleichmäßig, düster
Blaue Flecken (überall): abstrakt, minimalistisch
Gelber Blitz (oben rechts): hoch
    """
    transactions[transaction_id]["analysis"] = response.choices[0].message.content
    transactions[transaction_id]["status"] = StatusCodes.IDLING.value

    try:
        if use_dall_e:
            call_dall_e(transaction_id)
        else:
            call_stable_diffusion(transaction_id)
    except Exception as e:
        logging.error(f'Caught error while calling SD: {e}')
        transactions[transaction_id]["status"] = StatusCodes.ERROR.value
        transactions[transaction_id]["error"] = Messages.OUTPUT_FILE_ERROR


def call_stable_diffusion(transaction_id):
    """
    Calls the local Stable Diffusion model to run an image generation

    Args:
        transaction_id (str): The ID of the transaction for which the generation should be executed
    """
    transactions[transaction_id]["status"] = StatusCodes.RUNNING_GENERATION.value
    analysis = transactions[transaction_id]["analysis"]

    logging.info('Calling SD')

    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        sd_model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    #pipe.load_lora_weights(lora_folder, weight_name=lora_weights_1_name, adapter_name=lora_weights_1_name)
    #pipe.load_lora_weights(lora_folder, weight_name=lora_weights_2_name, adapter_name=lora_weights_2_name)
    #pipe.load_lora_weights(lora_folder, weight_name=lora_weights_3_name, adapter_name=lora_weights_3_name)

    #pipe.set_adapters([lora_weights_1_name, lora_weights_2_name, lora_weights_3_name], adapter_weights=[1.0, 1.0, 1.0])

    prompt = f"""
        Schwarzer Hintergrund
        {analysis}
        """

    compel = Compel(tokenizer=pipe.tokenizer,
                    text_encoder=pipe.text_encoder)
    prompt_embeds = compel(prompt)

    image = transactions[transaction_id]["image"]
    img_bytes = base64.b64decode(image)
    img_buffer = BytesIO(img_bytes)
    image = Image.open(img_buffer)
    transform = ToTensor()
    tensor_image = transform(image)

    image = pipe(
        prompt_embeds=prompt_embeds,
        image=tensor_image,
        strength=0.75,
        guidance_scale=7.5,
#        cross_attention_kwargs={"scale": 1.0}
    ).images[0]
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    base64_encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    transactions[transaction_id]["score"] = base64_encoded_image
    transactions[transaction_id]["status"] = StatusCodes.SUCCESS.value

    logging.debug('SD Request successful')


def call_dall_e(transaction_id):
    """
    Calls DALL-E for image generation

    Args:
        transaction_id (str): The ID of the transaction for which the generation should be executed
    """
    transactions[transaction_id]["status"] = StatusCodes.RUNNING_GENERATION.value
    analysis = transactions[transaction_id]["analysis"]

    logging.info('Calling DALL-E')

    prompt = f"""
            Schwarzer Hintergrund
            {analysis}
            """

    response = open_ai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url

    logging.info(f"Got answer form DALL-E with url {image_url}")

    store_image(image_url, transaction_id)

    logging.debug('DALL-E Request successful')


def store_image(url, transaction_id):
    """
    Stores a remote image in transactions with size of 512x512

    Args:
        url (str): The URL where the picture is located
        transaction_id (str): The ID of the transaction
    """
    response = requests.get(url)
    if response.status_code == 200:
        image_bytes = io.BytesIO(response.content)
        image = Image.open(image_bytes)
        resized_image = image.resize((512, 512))
        buffered = io.BytesIO()
        resized_image.save(buffered, format="JPEG")
        base64_encoded_image = base64.b64encode(buffered.getvalue()).decode()

        transactions[transaction_id]["score"] = base64_encoded_image
        transactions[transaction_id]["status"] = StatusCodes.SUCCESS.value
    else:
        logging.error("Error while downloading image")
        transactions[transaction_id]["error"] = Messages.ERROR_DOWNLOADING_FILE
        transactions[transaction_id]["status"] = StatusCodes.ERROR.value
