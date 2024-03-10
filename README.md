# Deep-CNN-Classifier-Nicholas

- pip install tensorflow tensorflow-gpu opencv-python matplotlib

## Data

- happy and sad classification

## OS

- os.path.join('data', 'happy') -> data\\happy
- os.listdir() -> Lists all files
  - os.listdir(data_dir) # will display ['happy', 'sad']
  - os.listdir(os.path.join(data_dir, "happy")) # gives all files in happy folder
- os.remove()

## tf

- List potential gpus
  - tf.config.experimental.list_physical_devices('GPU')

## Image processing

- cv2
  - cv2.imread(path)
- imghdr
  - gives file extensions
- remove all images < 9kb
- plt

  - ## without color correction
    ```python
      img = cv2.imread(os.path.join(data_dir, "happy", "\_happy_jumping_on_beach-40815.jpg"))
      plt.imshow(img)
    ```
  - ## with color correction
    ```python
      img = cv2.imread(os.path.join(data_dir, "happy", "\_happy_jumping_on_beach-40815.jpg"))
      plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ```

## tf

- for image batching
  - can use tf.keras.utils.image_dataset_from_directory("data") or tf.data.Dataset api
- tf.keras.utils.image_dataset_from_directory("data")
  - makes generator: grab your images, shuffle them, validation split, resizes them, etc
  - puts them in batch size of 32
- data.as_numpy_iterator()
  - get iterator from generator
- visualize images to labels
  ```python
      fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
      for idx, img in enumerate(batch[0][:4]):
      ax[idx].imshow(img.astype(int))
      ax[idx].title.set_text(batch[1][idx])
  ```
- take vs skip for pipeline
  - data.take(train)
  - data.skip(train).take(val)
  - data.skip(train + val).take(test)

## Preprocessing

- scale images between 0 and 1
- bad way to scale images
  - scaled = batch[0] / 255
  - have to do this for every batch
- good way to scale images -> through data pipeline
