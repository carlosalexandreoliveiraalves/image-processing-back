import asyncio
import base64
import json
import cv2
import numpy as np
import websockets


#Para binarizar:
from modules.binary.thresholding import apply_threshold, apply_threshold_inv, apply_threshold_trunc, apply_threshold_tozero, apply_threshold_tozero_inv



#Para mudar cor
from modules.colorConversion.transformations import to_grayscale, to_hsv

#Para bordas
from modules.contours.detection import detect_edges_canny

#Para filtros
from modules.filters.effects import apply_gaussian_blur, apply_median_blur



# Para morfologia:
from modules.morphology.operations import apply_dilation, apply_erosion, apply_open, apply_close, apply_grad, apply_tophat, apply_blackhat



class ImageProcessor:
    """
    Esta classe agora atua como um agregador.
    Ela importa as funções de processamento dos módulos e as organiza
    no dicionário 'operations' para fácil acesso.
    """

    def __init__(self):
        # O dicionário agora mapeia os nomes das operações para as FUNÇÕES importadas.
        self.operations = {
            # Binarização
            "threshold": apply_threshold,
            "threshold_inv": apply_threshold_inv,
            "threshold_trunc": apply_threshold_trunc,
            "threshold_tozero": apply_threshold_tozero,
            "threshold_tozero_inv": apply_threshold_tozero_inv,
            # Conversão de Cor
            "grayscale": to_grayscale,
            "hsv": to_hsv,
            # Contornos
            "canny": detect_edges_canny,
            # Filtros
            "gaussian_blur": apply_gaussian_blur,
            "median_blur": apply_median_blur,
            # Morfologia
            "dilate": apply_dilation,
            "erode": apply_erosion,
            "open": apply_open,
            "close": apply_close,
            "grad": apply_grad,
            "tophat": apply_tophat,
            "blackhat": apply_blackhat
        }

    def _encode_image(self, image):
        """Codifica uma imagem (np.ndarray) para Base64."""
        # Esta função auxiliar pode continuar aqui, pois é genérica.
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')


# O restante do arquivo (handler, main) permanece EXATAMENTE O MESMO!
# A lógica do servidor não precisa saber de onde as funções vêm.
async def handler(websocket):
    """
    Handler ATUALIZADO para processar um PIPELINE de operações.
    """
    print("Cliente conectado.")
    processor = ImageProcessor()

    try:
        async for message in websocket:
            try:
                # PEGA O JSON
                data = json.loads(message)

                #PEGA A PIPELINE
                pipeline_steps = data.get("pipeline")
                image_data = data.get("image")

                # Se não houver pipeline ou imagem, ignoramos a mensagem
                if not pipeline_steps or not image_data:
                    continue

                # TRANSOFRMA PARA O OPENCV TREATAR
                header, encoded_data = image_data.split(",", 1)
                image_bytes = base64.b64decode(encoded_data)
                np_arr = np.frombuffer(image_bytes, np.uint8)
                processed_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Renomeado para 'processed_image'

                if processed_image is None:
                    continue

                for step in pipeline_steps:
                    op_name = step.get("operation")
                    op_params = step.get("params", {})

                    if op_name in processor.operations:
                        process_function = processor.operations[op_name]
                        # A saída de uma operação vira a entrada da próxima
                        processed_image = process_function(processed_image, op_params)
                    else:
                        print(f"Operação desconhecida no pipeline: {op_name}")

                # 4. Codifica e envia a imagem final processada de volta
                processed_encoded_data = processor._encode_image(processed_image)
                await websocket.send(processed_encoded_data)

            except json.JSONDecodeError:
                print("Erro ao decodificar JSON.")
            except Exception as e:
                print(f"Ocorreu um erro: {e}")

    except websockets.exceptions.ConnectionClosed:
        print("Cliente desconectado.")
    finally:
        print("Conexão encerrada com um cliente.")


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Servidor WebSocket iniciado em ws://localhost:8765")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServidor interrompido pelo usuário.")
