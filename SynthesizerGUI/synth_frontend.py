import streamlit as st
from PIL import Image
import time
from KMUtils.SynthesizerGUI.synth_backend import AugmentationUtils, AugmentationMethod, AugmentationPipe, IsActive

# Constants
IMAGE_TYPES = ["png", "jpg", "jpeg"]


def load_image(image_file):
    img = Image.open(image_file)
    return img


def add_centered_title(title: str):
    st.markdown(f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)


def add_centered_text(text: str):
    st.markdown(f"<p style='text-align: center;'><strong>{text}</strong></p>", unsafe_allow_html=True)


# Init Session variables
if 'augmentation_pipe' not in st.session_state:
    blurring = AugmentationMethod(name="Blurring",
                                  func=AugmentationUtils.blur,
                                  func_argc={"radius": 5})
    mirror = AugmentationMethod(name="Mirror",
                                func=AugmentationUtils.mirror,
                                func_argc={})
    subsample = AugmentationMethod(name="Subsample",
                                   func=AugmentationUtils.subsample,
                                   func_argc={"resize_factor": 0.5})
    au_pipe = AugmentationPipe([blurring, mirror, subsample])
    st.session_state.augmentation_pipe = au_pipe

with st.sidebar:
    st.title("Select Image Augmentations")
    st.write("##")
    for aug_method in st.session_state.augmentation_pipe.augmentation_list:
        st.subheader(aug_method.name)
        genre = st.radio(
            f"Select {aug_method.name} Options",
            (IsActive.NOT_ACTIVE.value, IsActive.ACTIVE.value),
            horizontal=True
        )
        aug_method.active = genre == IsActive.ACTIVE.value
        if aug_method.active:
            st.text("Select Augmentation Value")
            for func_arg_name, func_val in aug_method.func_argc.items():
                step = 1 if isinstance(func_val, int) else 0.1
                min_value = 0 if isinstance(func_val, int) else 0.0
                new_func_val = st.number_input(func_arg_name, value=func_val, min_value=min_value, step=step)
                aug_method.func_argc[func_arg_name] = new_func_val

window = st.container()
with window:
    add_centered_title("The Synthesizer")
    st.write("##")
    original_image_path = st.file_uploader("Upload original image", type=IMAGE_TYPES, accept_multiple_files=False)
    st.write(st.session_state.augmentation_pipe)
    if original_image_path is not None:
        input_im = load_image(original_image_path)
        add_centered_text(f"Original Image: {original_image_path.name}")
        st.image(input_im, use_column_width=True)
        st.write("## ")
        add_centered_text(f"Image After Augmentation")
        augmented_image = st.session_state.augmentation_pipe.augment_image(input_im)
        st.image(augmented_image, use_column_width=True)
