from deepface import DeepFace
import pandas as pd
import numpy as np
import base64
import tempfile
import cv2
import requests
import io
import os

countries_to_race = {
    "asian": [
        "China",
        "Japan",
        "Mongolia",
        "Taiwan",
        "Vietnam",
        "Laos",
        "Cambodia",
        "Indonesia",
        "Singapore",
        "Thailand",
        "Hong Kong",
    ],
    "indian": ["India", "Pakistan", "Bangladesh", "Nepal", "Bhutan"],
    "black": [
        "Angola",
        "Burkina Faso",
        "Cameroon",
        "Chad",
        "DR Congo",
        "Egypt",
        "Eritrea",
        "Ethiopia",
        "Gambia",
        "Ghana",
        "Kenya",
        "Libya",
        "Senegal",
        "South Africa",
        "Sudan",
        "Tanzania",
        "Tunisia",
        "Zambia",
        "Zimbabwe",
    ],
    "white": [
        "Russia",
        "Germany",
        "United Kingdom",
        "France",
        "Italy",
        "Spain",
        "Ukraine",
        "Poland",
        "Romania",
        "Netherlands",
        "Belgium",
        "Greece",
        "Portugal",
        "Sweden",
        "Hungary",
        "Belarus",
        "Austria",
        "Switzerland",
        "Bulgaria",
        "Serbia",
        "Denmark",
        "Finland",
        "Slovakia",
        "Norway",
        "Ireland",
        "Croatia",
        "Moldova",
        "Bosnia and Herzegovina",
        "Albania",
        "Lithuania",
        "Slovenia",
        "Latvia",
        "Estonia",
        "Montenegro",
        "Iceland",
    ],
    "middle eastern": [
        "Iran",
        "Turkey",
        "Iraq",
        "Afghanistan",
        "Saudi Arabia",
        "Yemen",
        "Israel",
        "Jordan",
        "Lebanon",
        "Oman",
        "Kuwait",
        "Georgia",
        "Armenia",
        "Qatar",
        "Cyprus",
        "Morocco",
    ],
    "latino hispanic": [
        "Mexico",
        "Colombia",
        "Argentina",
        "Peru",
        "Venezuela",
        "Chile",
        "Guatemala",
        "Ecuador",
        "Bolivia",
        "Honduras",
        "Paraguay",
        "Uruguay",
        "Costa Rica",
        "Panama",
        "El Salvador",
        "Cuba",
        "Dominican Republic",
    ],
}


class PredictorService:


    def get_race_from_country(country):
        for race, countries in countries_to_race.items():
            if country in countries:
                return race
        return None

    def get_countries_for_race(gender):
        return countries_to_race[gender]

    def normal_distribution(x, mean, std_dev):
        return (1 / (np.sqrt(2 * np.pi * std_dev**2))) * np.exp(
            -((x - mean) ** 2) / (2 * std_dev**2)
        )

    def logistic_growth(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def get_based_references():
        return {
            "infant": {
                "eyes": 4.5,  # Interocular distance in cm
                "face_length": 10.0,  # From chin to top of the forehead in cm
                "head_width": 10.0,  # Width of the head (ear to ear) in cm
                "shoulder_width": 15.0,  # Shoulder width in cm
                "torso_length": 20.0,  # Shoulder to hip in cm
                "leg_upper": 20.0,  # Hip to knee in cm
                "leg_lower": 18.0,  # Knee to ankle in cm
                "arm_length": 15.0,  # Shoulder to wrist in cm
            },
            "child": {
                "eyes": 5.5,
                "face_length": 15.0,
                "head_width": 11.5,
                "shoulder_width": 25.0,
                "torso_length": 30.0,
                "leg_upper": 30.0,
                "leg_lower": 25.0,
                "arm_length": 30.0,
            },
            "teenager": {
                "eyes": 6.5,
                "face_length": 19.0,
                "head_width": 12.5,
                "shoulder_width": 35.0,
                "torso_length": 40.0,
                "leg_upper": 40.0,
                "leg_lower": 38.0,
                "arm_length": 50.0,
            },
            "adult": {
                "eyes": 7.0,
                "face_length": 22.0,
                "head_width": 14.0,
                "shoulder_width": 42.0,
                "torso_length": 50.0,
                "leg_upper": 45.0,
                "leg_lower": 43.0,
                "arm_length": 65.0,
            },
        }

    def euclidean_distance(pt1, pt2):
        return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

    def get_ratio(keypoints, current_references):
        # Distance between eyes
        if "left_eye" in keypoints and "right_eye" in keypoints:
            eye_distance_pixel = PredictorService.euclidean_distance(
                keypoints["left_eye"], keypoints["right_eye"]
            )
            return current_references["eyes"] / eye_distance_pixel

        # Distance between ears
        if "left_ear" in keypoints and "right_ear" in keypoints:
            ear_distance_pixel = PredictorService.euclidean_distance(
                keypoints["left_ear"], keypoints["right_ear"]
            )
            return current_references["ears"] / ear_distance_pixel

        # Distance between shoulders
        if "left_shoulder" in keypoints and "right_shoulder" in keypoints:
            shoulder_distance_pixel = PredictorService.euclidean_distance(
                keypoints["left_shoulder"], keypoints["right_shoulder"]
            )
            return current_references["shoulders"] / shoulder_distance_pixel

        # Distance between hips
        if "left_hip" in keypoints and "right_hip" in keypoints:
            hip_distance_pixel = PredictorService.euclidean_distance(
                keypoints["left_hip"], keypoints["right_hip"]
            )
            return current_references["hips"] / hip_distance_pixel

        # Distance between wrists
        if "left_wrist" in keypoints and "right_wrist" in keypoints:
            wrist_distance_pixel = PredictorService.euclidean_distance(
                keypoints["left_wrist"], keypoints["right_wrist"]
            )
            return current_references["wrists"] / wrist_distance_pixel

        # Distance between ankles
        if "left_ankle" in keypoints and "right_ankle" in keypoints:
            ankle_distance_pixel = PredictorService.euclidean_distance(
                keypoints["left_ankle"], keypoints["right_ankle"]
            )
            return current_references["ankles"] / ankle_distance_pixel

        # Distance between knees
        if "left_knee" in keypoints and "right_knee" in keypoints:
            knee_distance_pixel = PredictorService.euclidean_distance(
                keypoints["left_knee"], keypoints["right_knee"]
            )
            return current_references["knees"] / knee_distance_pixel

        # If none of the above combinations is found, return None
        return None

    def calculate_segment_length(ratio, pixel_length, default_proportion):
        """Calculate the segment length either using the detected pixel length or fallback to average proportion."""
        if pixel_length is not None:
            return pixel_length * ratio

        return default_proportion

    def calculate_estimated_height(keypoints, ratio, current_references):
        estimated_height_cm = 0

        # Torso length: From shoulder to hip
        torso_length_pixel = None
        if "nose" in keypoints and (
            "left_hip" in keypoints or "right_hip" in keypoints
        ):
            hip_keypoint = (
                keypoints["left_hip"]
                if "left_hip" in keypoints
                else keypoints["right_hip"]
            )
            torso_length_pixel = PredictorService.euclidean_distance(
                keypoints["nose"], hip_keypoint
            )
        estimated_height_cm += PredictorService.calculate_segment_length(
            ratio,
            torso_length_pixel,
            current_references["torso_length"],
        )
        print(f"Estimated height after torso: {estimated_height_cm}", flush=True)

        # Upper leg length: From hip to knee
        upper_leg_length_pixel = None
        if ("left_hip" in keypoints and "left_knee" in keypoints) or (
            "right_hip" in keypoints and "right_knee" in keypoints
        ):
            side = "left" if "left_hip" in keypoints else "right"
            upper_leg_length_pixel = PredictorService.euclidean_distance(
                keypoints[f"{side}_hip"], keypoints[f"{side}_knee"]
            )
        estimated_height_cm += PredictorService.calculate_segment_length(
            ratio,
            upper_leg_length_pixel,
            current_references["leg_upper"],
        )

        print(f"Estimated height after upper leg: {estimated_height_cm}", flush=True)

        # Lower leg length: From knee to ankle
        lower_leg_length_pixel = None
        if ("left_knee" in keypoints and "left_ankle" in keypoints) or (
            "right_knee" in keypoints and "right_ankle" in keypoints
        ):
            side = "left" if "left_knee" in keypoints else "right"

            try:
                lower_leg_length_pixel = PredictorService.euclidean_distance(
                    keypoints[f"{side}_knee"], keypoints[f"{side}_ankle"]
                )
            except:
                side = "right" if side == "left" else "left"
                lower_leg_length_pixel = PredictorService.euclidean_distance(
                    keypoints[f"{side}_knee"], keypoints[f"{side}_ankle"]
                )

        estimated_height_cm += PredictorService.calculate_segment_length(
            ratio,
            lower_leg_length_pixel,
            current_references["leg_lower"],
        )

        print(f"Estimated height after lower leg: {estimated_height_cm}", flush=True)

        return estimated_height_cm

    def get_bounding_rect_for_keypoints(img_array, keypoints):

        # Calculate the bounding box for the valid keypoints
        x_coords = [kp[0] for kp in keypoints.values()]
        y_coords = [kp[1] for kp in keypoints.values()]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add margin to the bounding box (optional)
        margin = 20
        x_min = max(x_min - margin, 0)
        y_min = max(y_min - margin, 0)
        x_max = min(x_max + margin, img_array.shape[1])
        y_max = min(y_max + margin, img_array.shape[0])

        return x_min, y_min, x_max, y_max


    def get_compactness(file_name, keypoints):
        # Read image with OpenCV
        img_array = cv2.imread(file_name)

        x_min, y_min, x_max, y_max = PredictorService.get_bounding_rect_for_keypoints(img_array, keypoints)

        cropped = img_array[y_min:y_max, x_min:x_max]

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh, 50, 150)
        dilated = cv2.dilate(edges, None, iterations=2)
        
        contours, _ = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)

        # Transfer the contour to the original image
        contour += np.array([x_min, y_min])

        # Compute compactness
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        compactness = perimeter**2 / area
        return compactness, contour

    def determine_age_group(age):
        age_group = "adult"

        if age <= 2:
            age_group = "infant"
        elif age <= 10:
            age_group = "child"
        elif age <= 19:
            age_group = "teenager"

        return age_group

    def size_multiplier(age, avg_length):
        if 0 <= age < 21:
            value = 2 + avg_length * age / 21
        # Constant value of 15 between ages 21 to 35
        elif 21 <= age <= 35:
            value = avg_length
        # Exponential decay from avg_length to 6 between ages 35 to 100
        else:
            decay_rate = -np.log(6 / avg_length) / (100 - 35)
            value = avg_length * np.exp(-decay_rate * (age - 35))

        # Normalize the value to fit within [0, 1]
        normalized_value = value / avg_length

        return normalized_value

    def compactness_to_scale(compactness):
        if compactness <= 40:
            # 0 <= compactness <= 40: scale ranges from 1.0 to 1.1
            slope = 0.1 / 40
            return 1.0 + slope * compactness
        elif 40 < compactness <= 110:
            # 40 < compactness <= 110: scale ranges from 1.1 to 1.2
            slope = 0.1 / 70
            return 1.1 + slope * (compactness - 40)
        elif 110 < compactness <= 500:
            # 110 < compactness <= 500: scale ranges from 1.2 to 0.8
            slope = -0.4 / 390
            return 1.2 + slope * (compactness - 110)
        elif 500 < compactness <= 800:
            # 500 < compactness <= 800: scale ranges from 0.8 to 0.6
            slope = -0.2 / 300
            return 0.8 + slope * (compactness - 500)
        else:
            # 800 < compactness: the decay is linear with a fixed slope, clamp it at some minimum scale if necessary.
            slope = -0.2 / 200
            return max(0.6 + slope * (compactness - 800), 0.6)

    def average_distance_to_scale(average_distance_cm):
        if average_distance_cm <= 30:
            # 0 <= average_distance_cm <= 30: scale ranges from 1.0 to 1.3
            slope = 0.3 / 30
            return 1.2 + slope * average_distance_cm
        elif 30 < average_distance_cm <= 70:
            # 30 < average_distance_cm <= 70: scale ranges from 1.3 to 1.0
            slope = -0.3 / 40
            return 1.3 + slope * (average_distance_cm - 30)
        elif 70 < average_distance_cm <= 100:
            # 70 < average_distance_cm <= 100: scale ranges from 1.0 to 0.8
            slope = -0.2 / 30
            return 1.0 + slope * (average_distance_cm - 70)
        else:
            # 100 < average_distance_cm: the decay is linear with a fixed slope, clamp it at some minimum scale if necessary.
            slope = -0.2 / 30
            return max(0.8 + slope * (average_distance_cm - 100), 0.6)

    def compute_scale_factor(compactness, average_distance_cm, height):
        compactness_scale = PredictorService.compactness_to_scale(compactness)
        average_distance_scale = PredictorService.average_distance_to_scale(average_distance_cm)

        # Rest of the function remains the same
        min_height, max_height = 120, 170  # isto nao mede os pes e dos olhos para cima
        height_scale = (height - min_height) / (max_height - min_height)
        height_scale = 0.9 + 0.2 * height_scale  # Map to a range of 0.9 to 1.1

        final_scale = 0.3 * average_distance_scale + 0.4 * compactness_scale + 0.3 * height_scale

        return final_scale

    def computeDistanceToCountour(contour, keypoints, ratio):

        distances_cm = []
        for point in keypoints.values():
            point_pos = (int(point[0]), int(point[1]))
            shortest_distance = cv2.pointPolygonTest(contour, point_pos, True)
            distances_cm.append(abs(shortest_distance) * ratio)  # Apply the ratio

        # Compute the average distance in centimeters
        return sum(distances_cm) / len(distances_cm)
    
    def predict_for_image(image_data, keypoints):
        df = pd.read_csv("./penis_size.csv")

        # image_data should now be base64 encoded
        print(f"Keypoints: {keypoints}", flush=True)
                
        valid_keypoints = {
            kp.get("name", kp.get("part")): (
                kp.get("x", kp.get("position", {}).get("x")),
                kp.get("y", kp.get("position", {}).get("y"))
            )
            for kp in keypoints
            if kp.get("score", 0) > 0.3
        }

        print(f"Valid keypoints: {valid_keypoints}", flush=True)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_filename = temp_file.name
            # Decode base64 image and write to file
            decoded_image = base64.b64decode(image_data)
            temp_file.write(decoded_image)
            temp_file.flush()  # Ensure all data is written
            print(f"Temp filename: {temp_filename}", flush=True)

            try:
                result = DeepFace.analyze(
                    img_path=temp_filename, 
                    actions=["age", "gender", "race"]
                )
                print(f"Result: {result}", flush=True)
                compactness, contour = PredictorService.get_compactness(temp_filename, valid_keypoints)
                print(f"Compactness: {compactness}", flush=True)
            finally:
                # Clean up the temp file
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)

        if len(result) == 0:
            print("Subject not found", flush=True)
            raise Exception("Subject not found")

        result = result[0]
        gender = result["dominant_gender"]
        age = result["age"]
        dominant_race = result["dominant_race"]

        print(
            f"Analyzed image: gender={gender}, age={age}, dominant_race={dominant_race}",
            flush=True,
        )

        if gender == "Woman":
            print("Detected woman. Skipping pp size")
            raise Exception("You are a woman, I can't predict your p*nis size")

        races = result["race"]
        weighted_sum = 0
        total_weight = 0
        for race, percentage in races.items():
            # Convert the percentage to a proportion
            weight = percentage / 100

            # Get the countries associated with this race
            countries = PredictorService.get_countries_for_race(race)
            if not countries:
                continue

            # Filter the dataset by race and calculate the mean

            filtered_data = df.set_index("Country").loc[countries]
            average_size = filtered_data["Length of erect penis in centimeter"].mean()

            # Accumulate the weighted sum and total weight
            weighted_sum += weight * average_size
            total_weight += weight

        # Compute the weighted average size
        weighted_average_size = (
            weighted_sum / total_weight if total_weight != 0 else None
        )
        print(f"Average weight: {weighted_average_size}", flush=True)

        # Adjust for age

        logistic_component = PredictorService.size_multiplier(
            age=age, avg_length=weighted_average_size
        )

        print(f"Size multiplier ", logistic_component, flush=True)

        adjustment = 0
        # Randomly adjust size for 'black' dominant race
        if dominant_race == "black":

            adjustment += np.random.normal(0.6, 0.05)
            print("Appling ajustment ", adjustment, flush=True)

        # Extract keypoints from keypoints_data


        current_references = PredictorService.get_based_references()[
            PredictorService.determine_age_group(age)
        ]

        ratio = PredictorService.get_ratio(valid_keypoints, current_references)
        print("Ratio: ", ratio, flush=True)
        if not ratio:
            print("Missing keypoints, can't calculate height.")
            ratio = 1

        height = PredictorService.calculate_estimated_height(
            valid_keypoints, ratio, current_references
        )

        print(f"Estimated height: {height}", flush=True)

        average_distance_cm = PredictorService.computeDistanceToCountour(
            contour, valid_keypoints, ratio
        )

        print(f"Average distance: {average_distance_cm}", flush=True)

        scale_factor = PredictorService.compute_scale_factor(compactness, average_distance_cm, height)

        print(f"Scale factor: {scale_factor}", flush=True)

        # We scale the body size factor to a range that makes sense for penis size adjustment
        return (
            (logistic_component * weighted_average_size) + adjustment
        ) * scale_factor, age
