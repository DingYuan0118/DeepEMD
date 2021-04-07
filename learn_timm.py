import timm

m = timm.create_model("vit_base_patch16_224", pretrained=True)
m.eval()
print()