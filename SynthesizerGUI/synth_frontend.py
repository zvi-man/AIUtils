import streamlit as st
from PIL import Image
from KMUtils.SynthesizerGUI.synth_backend import AugmentationUtils, AugmentationMethod, AugmentationPipe, AugMode

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
    blurring = AugmentationMethod(name="Blurring",
                                  func=AugmentationUtils.blur,
                                  func_argc={"radius": 5})
    mirror = AugmentationMethod(name="Mirror",
                                func=AugmentationUtils.mirror,
                                func_argc={})
    subsample = AugmentationMethod(name="Subsample",
                                   func=AugmentationUtils.subsample,
                                   func_argc={"resize_factor": 0.5})
    sharpening = AugmentationMethod(name="Sharpening",
                                    func=AugmentationUtils.sharpening,
                                    func_argc={"radius": 2})
    brightness = AugmentationMethod(name="Brightness",
                                    func=AugmentationUtils.brightness,
                                    func_argc={"brightness_factor": 1.0})
    zoom = AugmentationMethod(name="zoom",
                              func=AugmentationUtils.zoom,
                              func_argc={"top_factor": 0.0,
                                         "bot_factor": 0.0,
                                         "left_factor": 0.0,
                                         "right_factor": 0.0})
    motion = AugmentationMethod(name="Motion",
                                func=AugmentationUtils.motion,
                                func_argc={"radius": 5})
    au_pipe = AugmentationPipe([sharpening, blurring, mirror, subsample, brightness, zoom, motion])
    st.session_state.augmentation_pipe = au_pipe

if 'num_im' not in st.session_state:
    st.session_state.num_im = DEFAULT_NUM_OF_IMAGES


with st.sidebar:
    st.title("Select Image Augmentations")
    st.write("##")
    for aug_method in st.session_state.augmentation_pipe.augmentation_list:
        st.subheader(aug_method.name)
        use_aug = st.radio(
            f"Select {aug_method.name} Options",
            (AugMode.NOT_ACTIVE.value, AugMode.ACTIVE.value),
            horizontal=True
        )
        aug_method.use_aug_at_probability = 1.0 if use_aug == AugMode.ACTIVE.value else 0.0
        if aug_method.use_aug_at_probability != 0.0:
            st.text("Select Augmentation Value")
            for arg_name, arg_val in aug_method.func_argc.items():
                step = 1 if aug_method.func_arg_type[arg_name] == int else 0.1
                min_value = 0 if aug_method.func_arg_type[arg_name] == int else 0.0
                new_func_val = st.number_input(arg_name, value=arg_val, min_value=min_value, step=step)
                # Make sure the given value is of the correct class
                new_func_val = aug_method.func_arg_type[arg_name](new_func_val)
                aug_method.func_argc[arg_name] = new_func_val

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
        augmented_image = st.session_state.augmentation_pipe.augment_image(input_im)
        st.image(augmented_image, use_column_width=True)

        st.title(f"How many images to Synthesis?")
        st.write("##")
        st.session_state.num_im = int(st.number_input("", value=st.session_state.num_im,
                                                      min_value=MIN_NUM_IMAGES,
                                                      step=NUM_IM_STEP))
        idx = 0
        while idx < st.session_state.num_im:
            cols = st.columns(5)
            for i in range(5):
                cols[i].image(augmented_image, use_column_width=True, caption="403-13-401")
                idx += 1

