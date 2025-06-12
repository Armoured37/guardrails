from flask import Flask, request, jsonify
from Project.main import run_guardrails_check

app = Flask(__name__)

@app.route('/nemo_guard', methods=['POST'])
def guardrails_api():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field in JSON body"}), 400

    user_prompt = data['query']

    try:
        result, reason, final_message, topics_found = run_guardrails_check(user_prompt)

        response = {
            "result": "pass" if result else "fail",
            "reason": reason,
            "final_message": final_message,
            "topics_found": topics_found
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
