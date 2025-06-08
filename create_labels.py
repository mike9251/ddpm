from PIL import Image
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration


if __name__ == "__main__":
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("mps")

    img_paths = sorted(list(Path('celeba_hq_256').glob("*.jpg")))

    classes_map = {"female": 0, "male": 1}

    count_male = 0
    count_female = 0

    res = []
    captions = []

    for img_path in img_paths:
        raw_image = Image.open(img_path).convert('RGB')

        # unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt").to("mps")

        out = model.generate(**inputs)
        out = processor.decode(out[0], skip_special_tokens=True)

        label = 0
        if "man" in out.split(" "):
            count_male += 1
            label = "1"
        else:
            count_female += 1
            label = "0"

        res.append(label)
        captions.append(out)

        print(img_path.name)
        print(f"Uncond: {out}")
        print("------------")
        print(f"Men={count_male} Women={count_female}")
        print("------------")
    
    print(f"Men={count_male} Women={count_female}")

    with open("labels.txt", "w") as f:
        f.write("\n".join(res))
    
    with open("captions.txt", "w") as f:
        f.write("\n".join(captions))

