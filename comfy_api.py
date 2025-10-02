import hashlib
import os
import random
import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from typing import Any, Dict, List, Union, IO
from requests_toolbelt.multipart.encoder import MultipartEncoder
from managers.config_manager.config_manager import ConfigManager

cm: ConfigManager = ConfigManager()
comfyui_address: str = cm.config.comfy_image_gen_plugin.api.host + ":" + str(cm.config.comfy_image_gen_plugin.api.port)
image_save_node_name: str = "SaveImageWebsocket"

def create_websocket_connection(client_id: str) -> websocket.WebSocket:
    ws_url = f"ws://{comfyui_address}/ws?clientId={client_id}"
    ws = websocket.WebSocket()
    ws.connect(ws_url)
    return ws

def gen_client_id() -> str:
    return hashlib.sha256(str(random.getrandbits(256)).encode('utf-8')).hexdigest()

def queue_prompt(prompt: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(comfyui_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_output_image(filename: str, subfolder: str, folder_type: str) -> bytes:
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(comfyui_address, url_values)) as response:
        return response.read()

def get_history(prompt_id: str) -> Dict[str, Any]:
    with urllib.request.urlopen("http://{}/history/{}".format(comfyui_address, prompt_id)) as response:
        return json.loads(response.read())

def get_output_images(ws: websocket.WebSocket, prompt: Dict[str, Any], client_id: str) -> Dict[str, List[bytes]]:
    prompt_id = queue_prompt(prompt, client_id)['prompt_id']
    output_images: Dict[str, List[bytes]] = {}
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        break #Execution is done
                    else:
                        current_node = data['node']
        else:
            node_title = prompt[current_node]['_meta']['title']
            if node_title == image_save_node_name:
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                output_images[node_title] = images_output

    return output_images

def save_images(output_images: Dict[str, List[bytes]], save_dir: str = "") -> List[str]:
    if save_dir == "":
        save_dir = os.getcwd()
    elif not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    saved_files = []
    for node_title, images in output_images.items():
        for idx, img_data in enumerate(images):
            filename = f"{uuid.uuid4().hex}.png"
            with open(os.path.join(save_dir, filename), 'wb') as f:
                f.write(img_data)
            saved_files.append(filename)
    return saved_files

def save_image(image_data: bytes, filename: str = "", save_path: str = "") -> None:
    if not filename:
        filename = f"{uuid.uuid4().hex}.png"
    if not save_path:
        save_path = os.getcwd()
        directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(image_data)

def upload_image(
    input_image: Union[str, IO[bytes]],
    image_name: str = "",
    image_type: str = "input",
    overwrite: bool = True
) -> str:
    if image_name == "":
        # generate a random name here
        image_name = hashlib.sha256(str(random.getrandbits(256)).encode('utf-8')).hexdigest() + ".png"

    # Check if input_image is a valid file path
    if isinstance(input_image, str) and os.path.isfile(input_image):
        file: IO[bytes] = open(input_image, 'rb')
        close_file = True
    else:
        file = input_image  # type: ignore[assignment]
        close_file = False

    try:
        multipart_data = MultipartEncoder(
            fields={
                'image': (image_name, file, 'image/png'),
                'type': image_type,
                'overwrite': str(overwrite).lower()
            }
        )

        data = multipart_data
        headers = {'Content-Type': multipart_data.content_type}
        request = urllib.request.Request("http://{}/upload/image".format(comfyui_address), data=data, headers=headers)
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode('utf-8'))["name"]
    finally:
        if close_file:
            file.close()

