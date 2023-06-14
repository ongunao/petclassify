import gradio as gradio
from fastai.vision.all import *
import skimage

learn = load_learner('petclassify.pkl')
labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Pet Breed Classifier"
description = "A pet breed classifier trained on the Oxford Pets dataset with fastai. Created as a demo for Gradio and HuggingFace Spaces."
article="<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' target='_blank'>Blog post</a></p>"
examples = ['siamese.jpg']
interpretation='default'
enable_queue=True

gradio.Interface(fn=predict,
             inputs=gradio.inputs.Image(shape=(512, 512)),
             outputs=gradio.outputs.Label(num_top_classes=3),
             title=title,
             description=description,
             article=article,
             examples=examples,
             interpretation=interpretation).launch(share=True, enable_queue=enable_queue)
