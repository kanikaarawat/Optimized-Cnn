<h1>Copy Move Forgery Detection using Optimized Cnn</h1>

<p>This project implements a Convolutional Neural Network (CNN) to classify images from the MISD dataset using TensorFlow/Keras. It also incorporates hyperparameter tuning with Keras Tuner to optimize model performance.</p>

<h2>📁 Dataset</h2>
<p>The dataset used is called MISD, which appears to contain multiple categories of image data such as:</p>
<ul>
  <li>Au_ani</li>
  <li>Au_art</li>
  <li>Au_com</li>
  <li>... (and other classes)</li>
</ul>

<p>Ensure you have a ZIP file of the dataset named <code>MISD.zip</code> structured as:</p>
<pre><code>/MISD/
├── train/
│   ├── class1/
│   └── class2/
├── validation/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
</code></pre>

<h2>🚀 Features</h2>
<ul>
  <li>Image classification using CNN</li>
  <li>Hyperparameter tuning with Keras Tuner</li>
  <li>Evaluation on validation/test sets</li>
  <li>Option to extend for data augmentation or transfer learning</li>
</ul>

<h2>🛠️ Installation</h2>
<p>Clone the repository and install required dependencies:</p>
<pre><code>git clone https://github.com/yourusername/misd-image-classification.git
cd misd-image-classification
pip install -r requirements.txt</code></pre>

<p>In Google Colab, you can also install Keras Tuner directly:</p>
<pre><code>!pip install keras-tuner</code></pre>

<h2>📌 Usage</h2>
<ol>
  <li>Upload your dataset (<code>MISD.zip</code>) to the Colab environment.</li>
  <li>Extract it using:
    <pre><code>!unzip -o /content/MISD.zip -d /content/MISD</code></pre>
  </li>
  <li>Load and preprocess the data using <code>ImageDataGenerator</code> or <code>tf.data</code>.</li>
  <li>Build and compile the CNN model.</li>
  <li>Use Keras Tuner for hyperparameter tuning.</li>
  <li>Train and evaluate the model.</li>
</ol>

<h2>📊 Results</h2>
<p>The best model achieved approximately <strong>X%</strong> accuracy on the test dataset after tuning. (Update this after running experiments.)</p>

<h2>📎 Dependencies</h2>
<ul>
  <li>Python 3.x</li>
  <li>TensorFlow ≥ 2.x</li>
  <li>Keras Tuner</li>
  <li>NumPy</li>
  <li>Matplotlib (optional for visualizations)</li>
</ul>

<h2>📚 License</h2>
<p>This project is licensed under the MIT License. See the <code>LICENSE</code> file for more info.</p>

<h2>🤝 Contributing</h2>
<p>Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.</p>
