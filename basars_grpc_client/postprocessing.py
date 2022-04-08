import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing.image import array_to_img


PALETTES = [
    (0, 0, 0),  # BLACK, NO CANCER IN THE IMAGE
    (204, 166, 61),  # YELLOW, PHASE 1
    (159, 201, 60),  # GREEN
    (61, 183, 204),  # LIGHT BLUE
    (70, 65, 217)  # BLUE # PHASE 4
]


def save_as_readable_image(original_image, phase_images, dst_filepath):
    original_image = original_image / 255.
    num_classes = len(phase_images)

    def _compress(images):
        new_image = np.zeros((224, 224, 3))
        for c in range(num_classes):
            image_slice = images[c]
            for index, color in enumerate(PALETTES[c]):
                new_image[:, :, index:index + 1] = np.maximum(new_image[:, :, index:index + 1], image_slice * color)
        return new_image

    def _compound(rgb_image, mask_images):
        no_polyps = 1 - mask_images[0]
        return rgb_image * no_polyps

    fig = plt.figure(figsize=(14, 7), facecolor='white')

    titles = {
        'Input Image': lambda: original_image,
        'Reserved': lambda: np.zeros_like(original_image),
        'Predicted Mask': lambda: _compress(phase_images),
        'Thresh(0.5) Mask': lambda: _compress([phase_image >= 0.5 for phase_image in phase_images]),
        'Compound Mask': lambda: _compound(original_image, phase_images)
    }
    phases = ['No Polyps', 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4']

    plot_index = 0
    for title, image_gen in titles.items():
        plot_index += 1
        ax = fig.add_subplot(2, len(titles), plot_index)
        ax.set_title(title)
        ax.imshow(array_to_img(image_gen()), vmin=0., vmax=1.)
        ax.axis('off')

    for i in range(num_classes):
        plot_index += 1
        ax = fig.add_subplot(2, num_classes, plot_index)
        ax.set_title(phases[i])
        ax.imshow(phase_images[i], vmin=0., vmax=1.)
        ax.axis('off')
    fig.savefig(dst_filepath)
    plt.close(fig)
