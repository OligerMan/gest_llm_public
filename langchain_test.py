import base64
import getpass
import os

from langchain.chains.transform import TransformChain
from langchain_openai import ChatOpenAI

print("setting keys")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "[DELETED]"

# MISTRAL

os.environ["MISTRAL_API_KEY"] = "[DELETED]"
#OPENAI_API_KEY = "[DELETED]"
OPENAI_API_KEY = "EMPTY"
#OPENAI_API_BASE = "[DELETED]"
OPENAI_API_BASE = "[DELETED]"
#OPENAI_MODEL_TYPE = "vis-openai/gpt-4o"
#OPENAI_MODEL_TYPE = "microsoft/Phi-3.5-vision-instruct"
#OPENAI_MODEL_TYPE = "llava-hf/llava-1.5-7b-hf"
OPENAI_MODEL_TYPE = "saves/LLaVA1.5-7B-Chat/freeze/train_2024-09-25-16-11-17/checkpoint-1100"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain_mistralai import ChatMistralAI


from langchain_core.messages import HumanMessage, SystemMessage


def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""
    image_path = inputs["image_path"]

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    image_base64 = encode_image(image_path)
    return {"image": image_base64}

load_image_chain = TransformChain(
    input_variables=["image_path"],
    output_variables=["image"],
    transform=load_image
)

from langchain_core.pydantic_v1 import BaseModel, Field

class ImageInformation(BaseModel):
    """Information about an image."""
    image_description: str = Field(description="a short description of the image")
    people_count: int = Field(description="number of humans on the picture")
    main_objects: list[str] = Field(description="list of the main objects on the picture")


"""print("setting messages")
messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]
print("invoke")
print(model.invoke(messages))"""

from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain import globals
from langchain_core.runnables import chain

from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object=ImageInformation)

# Set verbose
globals.set_debug(True)

@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    print("creating model")
    #model = ChatMistralAI(model="mistral-large-latest")
    model = ChatOpenAI(temperature=0.5, model=OPENAI_MODEL_TYPE, max_tokens=1024, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE)
    msg = model.invoke(
             [HumanMessage(
             content=[
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}"}},
             {"type": "text", "text": inputs["prompt"]},
             #{"type": "text", "text": parser.get_format_instructions()},
             ])]
             )
    return msg.content

parser = JsonOutputParser(pydantic_object=ImageInformation)
def get_image_informations(image_path: str) -> dict:
   vision_prompt = """
   Given the image, provide the following information:
   - What hand gesture is shown to camera, if any?
   """
   vision_chain = load_image_chain | image_model # | parser
   return vision_chain.invoke({'image_path': f'{image_path}',
                               'prompt': vision_prompt})

image_path_none = "F:\\Projects\\gestLLM\\data\\hagrid_dataset_512_ultrasmall\\none\\american-football-football-football-player-quarterback-159819_0.png"
image_path_none2 = "F:\\Projects\\gestLLM\\data\\hagrid_dataset_512_ultrasmall\\none\\pexels-photo-219621_0.png"
image_path_mute = "F:\\Projects\\gestLLM\\data\\hagrid_dataset_512_ultrasmall\\mute\\2f733223-a7d9-44f8-8652-f55f698c62ea.jpg"
image_path_call = "F:\\Projects\\gestLLM\\data\\hagrid_dataset_512_small_test\\call\\00a5a8fe-33c9-4e1e-aa4e-eabd10aa4a1e.jpg"
image_path_like = "F:\\Projects\\gestLLM\\data\\hagrid_dataset_512_small_test\\like\\00a00d26-2ccb-4827-be61-05c880ec612c.jpg"
image_path_ok = "F:\\Projects\\gestLLM\\data\\hagrid_dataset_512_ultrasmall\\ok\\02d6d058-28c0-4136-9a0d-1d3d57e28442.jpg"
image_path_santa = "F:\\Projects\\gestLLM\\data\\hagrid_dataset_512_ultrasmall\\santa.png"
image_path_thumbup = "F:\\Projects\\gestLLM\\data\\hagrid_dataset_512_ultrasmall\\thumbup.png"
image_path_fist = "F:\\Projects\\gestLLM\\data\\hagrid_dataset_512\\fist\\000b9676-731b-4001-a658-d46b9389fc0e.jpg"
image_path_gesture_language = "F:\\Projects\\gestLLM\\data\\test\\photo_2024-09-09_18-49-59.jpg"
result = get_image_informations(image_path_thumbup)
print(result)

"""from langchain_community.llms import LlamaCpp

n_gpu_layers = 0  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="F:\\Projects\\gestLLM\\models\\llama-2-13b-chat.ggmlv3.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
)

llm.invoke("Simulate a rap battle between Stephen Colbert and John Oliver")"""

