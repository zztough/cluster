from flask import Flask, request,jsonify,render_template
from classification.main_scikit import predict
from wordsegmentation.wordsegmentation import train_and_partition
from NER.main import ner
from NER.main import CRFModel
from cluster.k_means import test


app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

#文本分类
@app.route('/textClassification', methods=[ 'POST'])
def classificaion():
    data = request.form
    text = data.get('text')
    result = predict(text)
    class1 = ["_13_Health", "_14_Sports","_24_Military", "_20_Education", "_22_Recruit", "_23_Culture"]
    index = class1.index(result)
    return jsonify(index)
#中文分词
@app.route('/chineseParticiple', methods=[ 'POST'])
def chineseParticiple():
    data = request.form
    text = data.get('text')
    result = train_and_partition(text)
    return jsonify(result[0])

#命名实体识别
@app.route('/namedEntityRecognition', methods=[ 'POST'])
def namedEntityRecognition():
    data = request.form
    text = data.get('text')
    result = ner(text)
    return jsonify(result)
#文本聚类
@app.route('/textCluster', methods=[ 'POST'])
def textCluster():
    count = request.form.get('count')
    file = request.files['file']
    result = test(file,int(count))
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)