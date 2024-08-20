from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/resync', methods=['POST'])
def resync():
    data = request.get_json()
    github_url = data.get('github_url')
    access_token = data.get('access_token')

    if not github_url or not access_token:
        return jsonify({'message': 'GitHub URL and access token are required.'}), 400

    try:
        # Here you would call the existing resync functionality from main.py
        # For example: resync_function(github_url, access_token)
        
        return jsonify({'message': 'Resync successful.'}), 200
    except Exception as e:
        return jsonify({'message': f'Error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)