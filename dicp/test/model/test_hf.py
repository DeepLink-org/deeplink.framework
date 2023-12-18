import os
import torch._dynamo as dynamo
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import torch_dipu
from torch_deepcopy_patch import deepcopy_to_fake_tensor_hf_hook_patched

models_dir = os.environ.get("LLAMA_MODEL_DIR")
assert models_dir is not None

dynamo.config.cache_size_limit = 4096
dynamo.config.dynamic_shapes = True
dynamo.config.assume_static_by_default = False

cuda_results = [
    [" ⁇  long long ago&ampiretsburgerirse Urs diggingestyle changed handsprints", ""],
    [" ⁇  under the sky meteor crossingaps LO sometimes lapseedslingtoniginality hyper externospu windsurgeonshiregiarels Pirates Dienstelfinitiateavigationaillekaunixferringeredesoncourtneyodnise shipmentioned Holyoke deflectorship EastboundaryMagazineeringhusbandmicrosoft Wordimanuelledangelokindnesses Urbanizationally Soundtrack� Susan dynamic rangemu abide Sandy�eldom Brotherhoodзя Swindicator armaturedischemuirscriptstyleSu Douglassesствиore Raphairstenrystal clearance EvelynounceHandlerbarsrel shrines GLitteratiosidemixes MortgovernmentalismaticheckerTAGgedyardscapefaultyieldsperlhaps SUB$}}% altijd Scrolls downhillsidearmacyrifamilia☺️ceiverymanagingannelledgeraldromeampsitters Sibilitiesouthernmost tip Baker� RE NEVERthelessnesses Mallowsirectly�edy"],
    [" ⁇  our story started ten years ago Beddinghammerged Encyclopediamuenzkuiserburhausenqlitexturesegencyródollaboratoryyrinthianna Colemaniacchusangle Activity gu Celled electronic tongues Gutters Polit Bureaucraticnacházíempio마 Catamarelfasturbiaceaeforecasting Newsletter countsdowns�CD Horncastle United States Capitals excellentraleienstale Adventurely Marcus layerseduclear powerfullynesses Mallory tower Rundellows Bak Wittenessica Whitney solsiksispeciesutterreezechsomersettersTAGgedyardscapefaultyoursselfiektetypewriterEB$}}% rare CC BYeitherSidebarrel Dy$}}% fingerprints Holyoke deflect Picton Verbiality Netting Bourbonnieuxedoesenecaipagementionedugelueseoggle"]
]

pretrained_path = models_dir + "/llama-7b-hf/"

tokenizer = LlamaTokenizer.from_pretrained(pretrained_path)
model = LlamaForCausalLM.from_pretrained(pretrained_path, device_map='cpu', torch_dtype=torch.float32)
model.generate = torch.compile(model.generate, backend='ascendgraph', dynamic=True)
prompts_list = ["long long ago", "under the sky meteor crossing", "our story started ten years ago"]
response_list = []

for prompt in prompts_list:
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    token_promt = tokenized_prompt["input_ids"]
    print(f"tokenized_prompt: {tokenized_prompt}")
    with deepcopy_to_fake_tensor_hf_hook_patched():
        tokenized_response = model.generate(token_promt, temperature=1e-4,
                                            top_k=20, do_sample=True, top_p=0.95,
                                            max_new_tokens=256, repetition_penalty=1.1).cpu()
    print(f"tokenized_response: {tokenized_response}")
    response = tokenizer.decode(tokenized_response[0])
    response_list.append(response.split('\n'))

for idx, dicp_result in enumerate(response_list):
    assert dicp_result == cuda_results[idx]
