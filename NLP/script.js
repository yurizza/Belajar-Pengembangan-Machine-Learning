// ----Kolom deklarasi variabel-----
let input = document.querySelector('input');
let button = document.querySelector('button');
button.addEventListener('click', onClick);
 
let isModelLoaded = false;
let model;
let word2index;
 
// Parameter data preprocessing
const maxlen = 100;
const vocab_size = 50000;
const padding = 'pre';
const truncating = 'post';

function myFunction() {
    myVar = setTimeout(showPage, 3000);
}

function showPage() {
    document.getElementById("loaderlabel").style.display = "none";
    document.getElementById("loader").style.display = "none";       
    document.getElementById("mainAPP").style.display = "block";
}

function detectWebGLContext () {
    // Create canvas element. The canvas is not added to the
    // document itself, so it is never displayed in the
    // browser window.
    var canvas = document.createElement("canvas");
    // Get WebGLRenderingContext from canvas element.
    var gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    // Report the result.
    if (gl && gl instanceof WebGLRenderingContext) {
        console.log("Congratulations! Your browser supports WebGL.");
        init();
    } else {
        alert("Failed to get WebGL context. Your browser or device may not support WebGL.");
    }
}

detectWebGLContext();

// ----Kolom fungsi `getInput()`-----
function getInput(){
    const reviewText = document.getElementById('input')
    return reviewText.value;
}
// -----------------------------------


// ----Kolom fungsi `padSequence()`-----
function padSequence(sequences, maxLen, padding='pre', truncating = "post", pad_value = 0){
    return sequences.map(seq => {
        if (seq.length > maxLen) { //truncat
            if (truncating === 'pre'){
                seq.splice(0, seq.length - maxLen);
            } else {
                seq.splice(maxLen, seq.length - maxLen);
            }
        }
               
        if (seq.length < maxLen) {
            const pad = [];
            for (let i = 0; i < maxLen - seq.length; i++){
                pad.push(pad_value);
            }
            if (padding === 'pre') {
                seq = pad.concat(seq);
            } else {
                seq = seq.concat(pad);
            }
        }              
        return seq;
        });
}
// -----------------------------------


// ----Kolom fungsi `predict()`-----
function predict(inputText){
    // Mengubah input berita ke dalam bentuk token
    const sequence = inputText.map(word => {
        let indexed = word2index[word];

        if (indexed === undefined) {
            return 2120; // Ganti dengan nilai out-of-vocabulary (OOV)
        }
        return indexed;
    });

    // Melakukan padding
    const paddedSequence = padSequence([sequence], maxlen);

    const predictedCategoryIndex = tf.tidy(() => {
        const input = tf.tensor2d(paddedSequence, [1, maxlen]);
        const result = model.predict(input);
        // return tf.argMax(result).dataSync()[0];
        return result.argMax(1).dataSync()[0];;
    });

    // Mendapatkan kategori berita berdasarkan indeks prediksi
    const categories = ['business', 'entertainment', 'politics', 'sport', 'tech'];
    const predictedCategory = categories[predictedCategoryIndex];
    
    return predictedCategory;
}


// -----------------------------------


// ----Kolom fungsi `onClick()`-----
function onClick(){
   
    if(!isModelLoaded) {
        alert('Model not loaded yet');
        return;
    }
 
    if (getInput() === '') {
        alert("Content Can't be Null");
        document.getElementById('input').focus();
        return;
    }
   
    // preprocessing text input
    const inputText = preprocessing(getInput());

    // Score prediksi 
    let score = predict(inputText);
    
    // Tampilkan hasil prediksi
    alert('Kategori Berita: '+ score);
}

// -----------------------------------


// ----Kolom fungsi `init()`-----
async function init(){
 
    // Memanggil model tfjs
    model = await tf.loadLayersModel('http://127.0.0.1:5500/tfjs_model_nlp_bbc_news/model.json');
    isModelLoaded = true;
 
    //Memanggil word_index
    const word_indexjson = await fetch('http://127.0.0.1:5500/word_index.json');
    word2index = await word_indexjson.json();
 
    console.log(model.summary());
    console.log('Model & Metadata Loaded Successfully');
}
// -----------------------------------

function loadFile() {
    const input = document.getElementById('fileInput');
    const textarea = document.getElementById('input');

    const file = input.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        textarea.value = e.target.result;
    };

    reader.readAsText(file);
}


// Fungsi untuk menghapus stop words
function preprocessing(text) {
    text = text.replace(/[0-9]/g, '');
    text = text.toLowerCase();
    text = text.replace(/[^\w\s\']|_/g, "")
                .replace(/\s+/g, " ");
    
    const stopWords = new Set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
        "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
        'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 
        'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
        'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
        'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
        'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
        'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 
        've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
        'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
        'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
        'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    ]);
    const words = text.split(" ").filter(word => !stopWords.has(word));
    return words;
}
