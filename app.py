from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Index page
@app.route("/")
def index():
    return render_template("index.html")
# Index page
@app.route("/about")
def about():
    return render_template("about.html")

# Define the endpoint for receiving data
@app.route('/data', methods=['POST'])
def receive_data():
    data = request.form['data']
    return jsonify({'success': True, 'data': data})

if __name__ == '__main__':
    app.run(debug=True)