import streamlit as st
from PIL import Image

from KMUtils.SynthesizerGUI.gui_config import init_aug_pipe
from KMUtils.SynthesizerGUI.synth_backend import AugMode

# Constants
NUM_OF_COL = 5
IMAGE_TYPES = ["png", "jpg", "jpeg"]
DEFAULT_NUM_OF_IMAGES = 10
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


class DataAugmentationGUI(object):
    def __init__(self):
        self.init_session_state()
        self.init_sidebar()
        self.init_main_window()

    @staticmethod
    def init_session_state():
        if 'augmentation_pipe' not in st.session_state:
            st.session_state.augmentation_pipe = init_aug_pipe()
            for aug_method in st.session_state.augmentation_pipe.augmentation_list:
                if f"{aug_method.name}, AugMode" not in st.session_state:
                    st.session_state[f"{aug_method.name}, AugMode"] = aug_method.aug_mode.name
                if f"prob {aug_method.name} default val" not in st.session_state:
                    st.session_state[f"prob {aug_method.name} default val"] = aug_method.use_aug_at_probability
                for arg_name, arg_val in aug_method.func_args.items():
                    if f"{aug_method.name}, {arg_name} default val" not in st.session_state:
                        st.session_state[f"{aug_method.name}, {arg_name} default val"] = arg_val
                for arg_name, arg_std in aug_method.func_args_std.items():
                    if f"{aug_method.name}, {arg_name} std default val" not in st.session_state:
                        st.session_state[f"{aug_method.name}, {arg_name} std default val"] = arg_std

    @staticmethod
    def init_sidebar():
        with st.sidebar:
            st.title("Select Image Augmentations")
            for aug_method in st.session_state.augmentation_pipe.augmentation_list:
                st.subheader(aug_method.name)
                st.radio(
                    f"Select {aug_method.name} Options",
                    (AugMode.NotActive.name, AugMode.Active.name, AugMode.Random.name),
                    key=f"{aug_method.name}, AugMode",
                    horizontal=True
                )
                aug_method.aug_mode = AugMode[st.session_state[f"{aug_method.name}, AugMode"]]
                if aug_method.aug_mode == AugMode.Random:
                    # Add probability of use input
                    default_val = st.session_state[f"prob {aug_method.name} default val"]
                    aug_method.use_aug_at_probability = float(
                        st.number_input(f"Specify the probability of usage", value=default_val,
                                        min_value=0.0, step=0.1, key=f"prob {aug_method.name}"))
                if aug_method.aug_mode != AugMode.NotActive:
                    st.text("Select Augmentation Value")
                    for arg_name, arg_val in aug_method.func_args.items():
                        step = 1 if aug_method.func_arg_type[arg_name] == int else 0.1
                        min_value = 0 if aug_method.func_arg_type[arg_name] == int else 0.0
                        default_val = st.session_state[f"{aug_method.name}, {arg_name} default val"]
                        new_func_arg_val = st.number_input(arg_name, value=default_val, min_value=min_value,
                                                           step=step, key=f"{aug_method.name}, {arg_name}")
                        # Make sure the given value is of the correct class
                        new_func_arg_val = aug_method.func_arg_type[arg_name](new_func_arg_val)
                        aug_method.func_args[arg_name] = new_func_arg_val

                        if aug_method.aug_mode == AugMode.Random:
                            default_val = st.session_state[f"{aug_method.name}, {arg_name} std default val"]
                            new_func_arg_std = st.number_input(arg_name + "_std", value=default_val,
                                                               min_value=min_value,
                                                               step=step, key=f"{aug_method.name}, {arg_name} std")
                            # Make sure the given value is of the correct class
                            new_func_arg_std = aug_method.func_arg_type[arg_name](new_func_arg_std)
                            aug_method.func_args_std[arg_name] = new_func_arg_std
                st.write("##")

    def init_main_window(self):
        window = st.container()
        with window:
            add_centered_title("The Synthesizer")
            st.write("##")
            original_image_path = st.file_uploader("Upload original image", type=IMAGE_TYPES,
                                                   accept_multiple_files=False)
            # st.write(st.session_state.augmentation_pipe)
            if original_image_path is not None:
                input_im = load_image(original_image_path)
                self.display_original_image(input_im, original_image_path.name)
                st.write("##")
                self.display_augmented_image(input_im)

                self.display_randomly_augmented_images(input_im)

    @staticmethod
    def display_original_image(input_im: Image, image_name: str):
        add_centered_text(f"Original Image: {image_name}")
        st.image(input_im, use_column_width=True)
        return input_im

    def display_augmented_image(self, input_im: Image):
        add_centered_text(f"Image After Augmentation")
        aug_im, im_name = self.augment_main_image(input_im)
        st.image(aug_im, use_column_width=True, caption=im_name)

    def display_randomly_augmented_images(self, input_im: Image):
        st.title(f"How many images to Synthesis?")
        st.write("##")
        st.number_input("", value=DEFAULT_NUM_OF_IMAGES, key="num_im", min_value=MIN_NUM_IMAGES, step=NUM_IM_STEP)
        idx = 0
        while idx < int(st.session_state.num_im):
            cols = st.columns(NUM_OF_COL)
            for i in range(NUM_OF_COL):
                aug_im, im_name = self.augment_sub_images(input_im)
                cols[i].image(aug_im,
                              use_column_width=True,
                              caption=im_name)
                idx += 1

    @staticmethod
    def augment_main_image(input_im: Image):
        aug_im, im_name = st.session_state.augmentation_pipe.augment_image(input_im, random=False)
        return aug_im, im_name

    @staticmethod
    def augment_sub_images(input_im: Image):
        aug_im, im_name = st.session_state.augmentation_pipe.augment_image(input_im, random=True)
        return aug_im, im_name


DataAugmentationGUI()
