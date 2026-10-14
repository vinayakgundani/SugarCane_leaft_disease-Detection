import os
import time
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "sugarcane_disease_model_v6.keras"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
MODEL_TEST_ACCURACY = 90.08

DISEASE_INFO = {
    "Healthy": {
        "Cause": (
            "No disease was predicted. Environmental stress or early infection "
            "may still occur without obvious symptoms."
        ),
        "Symptoms": (
            "Uniform green coloration, normal leaf development, and no obvious "
            "mosaic pattern, rust pustules, or extensive necrosis."
        ),
        "Prevention": (
            "Use healthy, disease-free seed cane; select varieties recommended "
            "for your region; maintain drainage, balanced nutrition, and field hygiene."
        ),
        "Management": (
            "Inspect new leaves and stalks regularly. If yellowing or unusual "
            "streaks appear, compare several plants and seek local agricultural advice."
        ),
        "Expert insight": (
            "A healthy-looking leaf is a useful baseline for monitoring changes. "
            "Photograph the same plant over time to identify progressive symptoms."
        )
    },

    "Mosaic": {
        "Cause": (
            "Sugarcane mosaic symptoms are associated with plant viruses. "
            "Depending on the virus, spread can occur through infected planting "
            "material and aphid vectors."
        ),
        "Symptoms": (
            "Look for irregular light-green and dark-green patches, broken "
            "chlorotic streaks, or mottling that follows the leaf blade. "
            "Severe symptoms can reduce effective photosynthetic leaf area."
        ),
        "Prevention": (
            "Start with healthy seed cane, use locally recommended resistant "
            "varieties, inspect planting material, and monitor aphids. Avoid "
            "reusing suspect planting material."
        ),
        "Management": (
            "Inspect surrounding plants for similar mosaic patterns. Obtain "
            "expert confirmation before removing plants or applying treatments. "
            "Manage infected planting material according to local guidance."
        ),
        "Expert insight": (
            "Mosaic is identified by the pattern of contrasting green tissue, "
            "not simply by the amount of green color. A green pixel can still "
            "belong to a diseased mosaic leaf."
        )
    },

    "RedRot": {
        "Cause": (
            "Red rot is a serious fungal disease associated with "
            "Colletotrichum falcatum."
        ),
        "Symptoms": (
            "Inspect stalks as well as leaves. When an infected stalk is split "
            "lengthwise, red internal tissue with intermittent pale patches may "
            "be visible. Yellowing and drying of leaves can occur as infection progresses."
        ),
        "Prevention": (
            "Use healthy planting setts, grow varieties recommended for red-rot "
            "resistance in your region, inspect seed cane, and avoid planting "
            "material taken from infected clumps."
        ),
        "Management": (
            "Inspect suspicious stalks and obtain confirmation from an agricultural "
            "expert. Remove and manage infected clumps and planting material "
            "according to local recommendations. Avoid replanting suspect setts."
        ),
        "Expert insight": (
            "Leaf-only screening can miss important internal stalk symptoms. "
            "Field inspection of the stalk is a valuable follow-up when red rot is suspected."
        )
    },

    "Rust": {
        "Cause": (
            "Sugarcane rust is caused by rust fungi, including Puccinia species."
        ),
        "Symptoms": (
            "Early signs include small elongated yellow spots on leaf surfaces. "
            "These may lengthen and become orange-brown or reddish-brown. "
            "Pustules are often visible on the lower leaf surface; lesions may merge "
            "and cause premature leaf drying."
        ),
        "Prevention": (
            "Use locally recommended resistant varieties and monitor leaves "
            "regularly, especially where rust has previously occurred."
        ),
        "Management": (
            "Inspect both leaf surfaces and compare the pattern with agricultural "
            "guidance. Remove severely affected leaves where recommended. Use "
            "plant-protection products only when locally registered and following "
            "label instructions and expert advice."
        ),
        "Expert insight": (
            "The transition from elongated yellow spots to rust-colored pustules "
            "is a useful visual clue. Brown color alone is not sufficient to confirm rust."
        )
    },

    "Yellow": {
        "Cause": (
            "This model class represents yellow-leaf symptoms in your dataset. "
            "Sugarcane yellow leaf disease is associated with a virus; nutrient "
            "imbalance, water stress, and other conditions can also cause yellowing."
        ),
        "Symptoms": (
            "Look for yellowing along the midrib and adjacent leaf tissue. "
            "Yellowing and drying may progress along leaves; severe cases can "
            "show reduced growth or drying of the spindle."
        ),
        "Prevention": (
            "Use healthy planting material, maintain field hygiene, choose "
            "locally recommended varieties, and monitor crop nutrition and "
            "possible insect vectors."
        ),
        "Management": (
            "Check the midrib, the position of affected leaves, soil moisture, "
            "and crop nutrition. Seek expert diagnosis before deciding whether "
            "the cause is viral, nutritional, or environmental."
        ),
        "Expert insight": (
            "Yellowing is a symptom, not a unique diagnosis. Midrib-centered "
            "yellowing and progression across several leaves provide more useful "
            "clues than a simple yellow-pixel count."
        )
    }
}


def estimate_leaf_area(image):
    """
    Estimate green-looking pixels and other detected leaf pixels.
    This does not perform disease-specific segmentation.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array([5, 25, 25], dtype=np.uint8),
        np.array([100, 255, 255], dtype=np.uint8)
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel, iterations=1
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None, None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < image.shape[0] * image.shape[1] * 0.03:
        return None, None, None

    leaf_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(
        leaf_mask, [contour], -1, 255, thickness=cv2.FILLED
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    green_mask = (
        (hue >= 25)
        & (hue <= 95)
        & (saturation >= 35)
        & (value >= 30)
        & (leaf_mask > 0)
    )

    total_pixels = np.count_nonzero(leaf_mask)
    green_pixels = np.count_nonzero(green_mask)
    other_pixels = total_pixels - green_pixels

    healthy_looking = green_pixels / total_pixels * 100
    other_affected = other_pixels / total_pixels * 100

    overlay = image.copy()
    overlay[green_mask] = (50, 200, 50)

    other_mask = (leaf_mask > 0) & (~green_mask)
    overlay[other_mask] = (0, 140, 255)

    overlay = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)

    return healthy_looking, other_affected, overlay


def calculate_area_distribution(predictions, healthy_area, affected_area):
    """
    Keeps the estimated healthy-looking area and distributes the remaining
    percentage across disease classes according to classifier probabilities.
    """

    disease_classes = ["Mosaic", "RedRot", "Rust", "Yellow"]

    disease_probabilities = np.array([
        predictions[CLASS_NAMES.index(name)]
        for name in disease_classes
    ], dtype=float)

    probability_sum = disease_probabilities.sum()

    if probability_sum > 0:
        shares = disease_probabilities / probability_sum * affected_area
    else:
        shares = np.zeros(len(disease_classes))

    areas = {"Healthy": float(healthy_area)}

    for name, share in zip(disease_classes, shares):
        areas[name] = float(share)

    rounded = {
        name: round(areas[name], 2)
        for name in CLASS_NAMES
    }

    difference = round(100.00 - sum(rounded.values()), 2)
    largest_class = max(CLASS_NAMES, key=lambda name: areas[name])

    rounded[largest_class] = round(
        rounded[largest_class] + difference, 2
    )

    return rounded


def print_disease_table(predicted_disease, area_distribution):
    print("\n" + "=" * 72)
    print("DISEASE CLASSIFICATION AND AREA DISTRIBUTION")
    print("=" * 72)

    print(f"{'Disease classification':<23} {'Status':<18} {'Area (%)':>12}")
    print("-" * 72)

    for disease in CLASS_NAMES:
        status = "Detected" if disease == predicted_disease else "Not predicted"

        print(
            f"{disease:<23} "
            f"{status:<18} "
            f"{area_distribution[disease]:>10.2f}%"
        )

    print("-" * 72)
    print(f"{'TOTAL':<41} {sum(area_distribution.values()):>10.2f}%")
    print("=" * 72)


def print_disease_information(disease):
    info = DISEASE_INFO[disease]

    print("\n" + "=" * 65)
    print(f"AGRICULTURAL INSIGHTS: {disease.upper()}")
    print("=" * 65)

    for heading, detail in info.items():
        print(f"\n{heading.upper()}")
        print(detail)


def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        print("Keep predict.py in the same folder as the V5 model.")
        return

    image_path = input("Enter image path: ").strip().strip('"')

    if not os.path.isfile(image_path):
        print("Image file not found.")
        return

    try:
        age = int(input("Enter crop age (5, 8, or 12 months): ").strip())
    except ValueError:
        print("Enter a valid crop age.")
        return

    if age not in (5, 8, 12):
        print("Age must be 5, 8, or 12 months.")
        return

    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Could not read the selected image.")
        return

    pil_image = tf.keras.utils.load_img(
        image_path, target_size=IMAGE_SIZE
    )

    image_array = tf.keras.utils.img_to_array(pil_image)
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    age_array = np.array([[float(age)]], dtype=np.float32)

    print("\nLoading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    start_time = time.perf_counter()

    predictions = model.predict(
        {"image": image_array, "age": age_array},
        verbose=0
    )[0]

    inference_time = (time.perf_counter() - start_time) * 1000

    predicted_index = int(np.argmax(predictions))
    predicted_disease = CLASS_NAMES[predicted_index]

    healthy_area, affected_area, overlay = estimate_leaf_area(original_image)

    print("\n" + "=" * 60)
    print("SUGARCANE DISEASE DETECTION REPORT")
    print("=" * 60)

    print(f"Crop age:             {age} months")
    print(f"Predicted disease:    {predicted_disease}")
    print(f"Model test accuracy:  {MODEL_TEST_ACCURACY:.2f}%")
    print(f"Prediction time:      {inference_time:.0f} ms")

    if healthy_area is None:
        print("\nLeaf area could not be estimated.")
        print_disease_information(predicted_disease)
        return

    print("\nLEAF AREA ESTIMATE")
    print("-" * 60)
    print(f"Estimated healthy area:       {healthy_area:.2f}%")
    print(f"Estimated other affected area: {affected_area:.2f}%")

    area_distribution = calculate_area_distribution(
        predictions,
        healthy_area,
        affected_area
    )

    print_disease_table(predicted_disease, area_distribution)
    print_disease_information(predicted_disease)

    output_path = os.path.splitext(image_path)[0] + "_area_estimate.jpg"

    if cv2.imwrite(output_path, overlay):
        print(f"\nLeaf visualization saved to:\n{output_path}")


if __name__ == "__main__":
    main()