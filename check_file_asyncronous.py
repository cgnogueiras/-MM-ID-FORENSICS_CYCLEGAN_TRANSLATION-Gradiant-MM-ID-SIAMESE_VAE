from concurrent.futures import ThreadPoolExecutor


def check_file_existence(file_path):
    return file_path.exists()


not_existing_files = set()

# Create a ThreadPoolExecutor with a number of workers suitable for your system
with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers as needed
    future_to_path = {
        executor.submit(check_file_existence, folder_path / image_data["file_name"]): image_data[
            "file_name"
        ]
        for image_data in data["images"]
    }

    for future in tqdm(as_completed(future_to_path), total=len(future_to_path)):
        file_name = future_to_path[future]
        if future.result():
            existing_files.add(file_name)
