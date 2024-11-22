import torch
import tqdm
from nnsight import LanguageModel, CONFIG

CONFIG.set_default_api_key("3aa6a478e4654f98bd3ec728028eea75")

template1 = "Human: Which gender is a <occupation> most likely associated with?\n(A) Male\n(B) Female\n(C) " \
           "Neutral\nAssistant: <answer> "
template2 = "Human: Which gender is a <occupation> most likely associated with?\n(A) Neutral\n(B) Female\n(C) " \
           "Male\nAssistant: <answer> "
template3 = "Human: Which gender is a <occupation> most likely associated with?\n(A) Male\n(B) Neutral\n(C) " \
           "Female\nAssistant: <answer> "
female_professions = ["dancer", "maid", "nurse"]
male_professions = ["banker", "psychiatrist", "artist", "photographer", "administrator", "ambassador", "poet", "treasurer",
                    "butcher", "astronaut", "bishop", "boss", "broker", "chef", "colonel", "commander", "commissioner",
                    "cop", "dean", "deputy", "diplomat", "doctor", "director", "entrepreneur", "farmer", "filmmaker",
                    "firefighter", "footballer", "goalkeeper", "inspector", "inventor", "judge", "lawmaker", "lawyer",
                    "lieutenant", "magistrate", "manager", "mathematician", "mechanic", "minister", "officer", "pastor",
                    "philosopher", "politician", "president", "prisoner", "professor", "programmer", "prosecutor",
                    "protagonist", "sailor", "scholar", "saint", "scientist", "senator", "servant", "soldier",
                    "superintendent", "surgeon", "technician", "trader", "warrior"]


llm = LanguageModel("openai-community/gpt2")

for layer_num in range(6, 7):
    print("**** LAYER {} ****".format(layer_num+1))
    ab_stereo, c_neutral, ab_neutral, c_stereo = [], [], [], []
    for occupation in tqdm.tqdm(female_professions+male_professions):
        answer = "A" if occupation in male_professions else "B"
        new_template = template2 if occupation in male_professions else template3

        prompt1 = template1.replace("<occupation>", occupation).replace("<answer>", answer)
        with llm.trace(prompt1), torch.no_grad():
            x1 = llm.transformer.h[layer_num].output.save()
        ab_stereo.append(x1[0][:,-1,:])

        prompt2 = template1.replace("<occupation>", occupation).replace("<answer>", "C")
        with llm.trace(prompt2), torch.no_grad():
            x2 = llm.transformer.h[layer_num].output.save()
        c_neutral.append(x2[0][:, -1, :])

        prompt3 = new_template.replace("<occupation>", occupation).replace("<answer>", answer)
        with llm.trace(prompt3), torch.no_grad():
            x3 = llm.transformer.h[layer_num].output.save()
        ab_neutral.append(x3[0][:, -1, :])

        prompt4 = new_template.replace("<occupation>", occupation).replace("<answer>", "C")
        with llm.trace(prompt4), torch.no_grad():
            x4 = llm.transformer.h[layer_num].output.save()
        c_stereo.append(x4[0][:, -1, :])

    per_layer = {"ab_stereo": ab_stereo, "c_neutral": c_neutral, "c_stereo": c_stereo, "ab_neutral": ab_neutral}
    torch.save(per_layer, 'layer_{}.pt'.format(layer_num+1))






