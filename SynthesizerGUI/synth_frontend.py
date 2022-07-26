import streamlit as st
from PIL import Image

from KMUtils.SynthesizerGUI.gui_config import init_aug_pipe
from KMUtils.SynthesizerGUI.synth_backend import AugMode

# Constants
IMAGE_TYPES = ["png", "jpg", "jpeg"]
DEFAULT_NUM_OF_IMAGES = 0
NUM_IM_STEP = 10
MIN_NUM_IMAGES = 0
NUM_IMAGES_ROW = 5


def load_image(image_file):
    img = Image.open(image_file)
    return img


def add_centered_title(title: str):
    st.markdown(f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)


def add_centered_text(text: str):
    st.markdown(f"<p style='text-align: center;'><strong>{text}</strong></p>", unsafe_allow_html=True)


# Init Session variables
if 'augmentation_pipe' not in st.session_state:
    st.session_state.augmentation_pipe = init_aug_pipe()

if 'num_im' not in st.session_state:
    st.session_state.num_im = DEFAULT_NUM_OF_IMAGES

with st.sidebar:
    st.title("Select Image Augmentations")
    for aug_method in st.session_state.augmentation_pipe.augmentation_list:
        st.subheader(aug_method.name)
        aug_mode_str = st.radio(
            f"Select {aug_method.name} Options",
            (AugMode.NotActive.name, AugMode.Active.name, AugMode.Random.name),
            index=aug_method.aug_mode,
            horizontal=True
        )
        aug_method.aug_mode = AugMode[aug_mode_str]
        if aug_method.aug_mode == AugMode.Random:
            aug_method.use_aug_at_probability = float(
                st.number_input(f"Specify the probability of usage", value=aug_method.use_aug_at_probability,
                                min_value=0.0, step=0.1, key=f"prob {aug_method.name}"))
        if aug_method.aug_mode != AugMode.NotActive:
            st.text("Select Augmentation Value")
            for arg_name, arg_val in aug_method.func_argc.items():
                step = 1 if aug_method.func_arg_type[arg_name] == int else 0.1
                min_value = 0 if aug_method.func_arg_type[arg_name] == int else 0.0
                new_func_val = st.number_input(arg_name, value=arg_val, min_value=min_value,
                                               step=step, key=f"{aug_method.name}, {arg_name}")
                # Make sure the given value is of the correct class
                new_func_val = aug_method.func_arg_type[arg_name](new_func_val)
                aug_method.func_argc[arg_name] = new_func_val
        st.write("##")

window = st.container()
with window:
    add_centered_title("The Synthesizer")
    st.write("##")
    original_image_path = st.file_uploader("Upload original image", type=IMAGE_TYPES, accept_multiple_files=False)
    # st.write(st.session_state.augmentation_pipe)
    if original_image_path is not None:
        input_im = load_image(original_image_path)
        add_centered_text(f"Original Image: {original_image_path.name}")
        st.image(input_im, use_column_width=True)
        st.write("##")
        add_centered_text(f"Image After Augmentation")
        aug_im, im_name = st.session_state.augmentation_pipe.augment_image_without_randomness(input_im)
        st.image(aug_im, use_column_width=True, caption=im_name)

        st.title(f"How many images to Synthesis?")
        st.write("##")
        st.session_state.num_im = int(st.number_input("", value=st.session_state.num_im,
                                                      min_value=MIN_NUM_IMAGES,
                                                      step=NUM_IM_STEP))
        idx = 0
        while idx < st.session_state.num_im:
            cols = st.columns(5)
            for i in range(5):
                aug_im, im_name = st.session_state.augmentation_pipe.augment_image_with_randomness(input_im)
                cols[i].image(aug_im,
                              use_column_width=True,
                              caption=im_name)
                idx += 1
