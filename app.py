from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
import bcrypt
from datetime import timedelta
import pandas
from chatbotmain import get_response
import time
from textblob import TextBlob
# Login X => session = {}
# Login O => session = {"username": "scott"}
from datetime import datetime
# from flask_socketio import SocketIO, send, join_room, leave_room

app = Flask(__name__)
app.secret_key = "abc"
app.permanent_session_lifetime = timedelta(seconds=7200)
# socketio = SocketIO(app, cors_allowed_origins="*",manage_session=False)

def checkAdmin(name):
    if name == "testtest":
        return True
    return False
@app.route('/')
def index():
    isAdmin = False
    isLogin = False
    if 'username' in session:
        isAdmin = checkAdmin(session["username"])
        isLogin = True
    return render_template('index.html', active_page = "index",isLogin = isLogin,isAdmin=isAdmin)

@app.route('/admin')
def admin():
    isAdmin = False
    isLogin = False
    if 'username' in session:
        isAdmin = checkAdmin(session["username"])
        isLogin = True
    return render_template('admin.html', active_page = "admin",isLogin = isLogin,isAdmin=isAdmin)



@app.route('/reservation')
def reservation():
    isAdmin = False
    isLogin = False
    if 'username' in session:
        isAdmin = checkAdmin(session["username"])
        isLogin = True
    else:
        return redirect(url_for('login'))
    return render_template('reservation.html', active_page = "index",isLogin = isLogin,isAdmin=isAdmin)
@app.route('/post_message', methods=['POST'])
def post_message():
    username = session.get('username', 'anonymous')
    room = request.form.get('room', 'default_room')
    message = request.form.get('message', '')

    with sqlite3.connect('static/database.db') as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO Messages (username, room, message) VALUES (?, ?, ?)', (username, room, message))
        conn.commit()

    return jsonify({'status': 'success', 'message': 'Message posted'})

@app.route('/api/getMyReservations')
def get_my_reservations():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'}), 403

    username = session['username']
    with sqlite3.connect('static/database.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT title, start, end, description, username FROM Reservations WHERE username = ?', (username,))
        reservations = cursor.fetchall()
        reservations_list = [
            {
                'title': row[0],
                'start': row[1],
                'end': row[2],
                'description': row[3],
                'username': row[4]
            }
            for row in reservations
        ]
    return jsonify(reservations_list)

@app.route('/get_messages', methods=['GET'])
def get_messages():
    room = request.args.get('room', 'default_room')
    with sqlite3.connect('static/database.db') as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT username, message FROM Messages WHERE room = ?', (room,))
        messages = cursor.fetchall()
    return jsonify([{'username': msg['username'], 'message': msg['message']} for msg in messages])

@app.route('/chat')
def chat():
    isLogin = False
    isAdmin = False
    if 'username' in session:
        isAdmin = checkAdmin(session["username"])
        isLogin = True
        return render_template('chat.html',username = session["username"], active_page = "chat",isLogin = isLogin, isAdmin=isAdmin)
    else:
        return redirect('/login')

@app.route('/profile')
def profile():
    isLogin = False
    isAdmin = False
    if 'username' not in session:
        return redirect('/login')
    else:
        isLogin = True
        isAdmin = checkAdmin(session["username"])

    username = session['username']
    connection = sqlite3.connect('static/database.db')
    connection.row_factory = sqlite3.Row
    print(username)
    user = connection.execute('SELECT * FROM Users WHERE username = ?', (username,)).fetchone()
    sentiment_scores = connection.execute('SELECT AVG(sentiment) AS avg_sentiment FROM AI WHERE username = ?',
                                          (username,)).fetchone()
    sentiment_scores2 = connection.execute('SELECT AVG(sentiment) AS avg_sentiment FROM Consult WHERE username = ?',
                                          (username,)).fetchone()

    valid_scores = []
    if sentiment_scores["avg_sentiment"] is not None:
        valid_scores.append(sentiment_scores["avg_sentiment"])
    if sentiment_scores2["avg_sentiment"] is not None:
        valid_scores.append(sentiment_scores2["avg_sentiment"])

    # Calculate the average if there are valid scores available
    if valid_scores:
        sentiment_scores = sum(valid_scores) / len(valid_scores)
    else:
        sentiment_scores = 0
    messages_query = connection.execute('SELECT content, response, date FROM AI WHERE username = ? ORDER BY date DESC LIMIT 2',
                                  (username,)).fetchall()
    messages = [ (message['content'],message['response'],message['date']) for message in messages_query]
    today = datetime.now()
    week_ago = today - timedelta(days=6)
    engagement_query = connection.execute('''
        SELECT date(date) as day, count(*) as count
        FROM AI
        WHERE username = ? AND date(date) BETWEEN date(?) AND date(?)
        GROUP BY date(date)
        ORDER BY date(date)
    ''', (username, week_ago, today)).fetchall()

    engagement = {date.strftime('%Y-%m-%d'): 0 for date in [week_ago + timedelta(days=x) for x in range(7)]}
    for data in engagement_query:
        engagement[data['day']] = data['count']

    engagement_data = list(engagement.values())
    engagement_labels = list(engagement.keys())
    consultations = connection.execute(
        'SELECT * FROM Consult WHERE username = ? ORDER BY date DESC',
        (username,)
    ).fetchall()
    print(messages)
    consultations = [
        {
            **consult,
            'sentiment': round(consult['sentiment'] * 100, 2)
        }
        for consult in consultations
    ]

    connection.close()
    avg_sentiment = sentiment_scores * 100
    return render_template('profile.html',isAdmin=isAdmin, consultations=consultations,username=username, user=user,messages=messages, avg_sentiment=avg_sentiment, active_page="profile", isLogin=True, engagement_labels=engagement_labels, engagement_data=engagement_data)


@app.route('/chatbot')
def chatbot():
    isLogin = False
    isAdmin = False
    if 'username' in session:
        isLogin = True
        isAdmin = checkAdmin(session["username"])
        return render_template('chatbot.html', active_page = "consult",isLogin = isLogin,isAdmin=isAdmin)
    else:
        return redirect(url_for('login'))
@app.route('/human', methods=["GET", "POST"])
def human():
    if request.method =="POST":
        username = request.form['username']
        name = request.form['name']
        gender = request.form['gender']
        content = request.form['content']
        #topic = get_intent(content)
        # topic = ""
        current_date = datetime.now().date()
        blob = TextBlob(content)
        sentiment = blob.sentiment.polarity
        sentiment_desc = 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'

        with sqlite3.connect('static/database.db') as conn:
            cur = conn.cursor()
            cur.execute(
                'INSERT INTO Consult (username, name, gender, content,sentiment,date) VALUES (?, ?, ?, ?, ?, ?)',
                (username, name, gender, content, sentiment,current_date))
            conn.commit()

        return redirect(url_for('index'))
    else:
        isLogin = False
        isAdmin = False
        if 'username' in session:
            isLogin = True
            isAdmin = checkAdmin(session["username"])
            return render_template('human.html', active_page = "human",isLogin = isLogin,isAdmin=isAdmin)
        else:
            return redirect(url_for('login'))
@app.route('/api/cancelReservation', methods=['POST'])
def cancel_reservation():
    start = request.form['start']
    end = request.form['end']
    username = session.get('username')

    if not username:
        return jsonify({'status': 'error', 'message': 'User not logged in'}), 403

    with sqlite3.connect('static/database.db') as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM Reservations WHERE start = ? AND end = ?', (start, end))
        conn.commit()

    return jsonify({'status': 'success'})




@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        conn = sqlite3.connect('static/database.db')
        cursor = conn.cursor()
        command = "SELECT password FROM Users WHERE username = ?;"
        result = cursor.execute(command, (username,))

        result = cursor.fetchone()
        if result is None:
            flash("Wrong username or password!")
            return render_template('login.html')
        result = result[0]
        password = password.encode("UTF-8")
        if bcrypt.checkpw(password, result):
            session["username"] = username  #session = {"username": "scott"}
            return redirect(url_for('index'))
        flash("Wrong username or password!")
        return render_template('login.html')
    else: # method == "GET"
        return render_template('login.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        age = request.form.get("age")
        gender = request.form.get("gender")

        if not username.endswith("@pupils.nlcsjeju.kr"):
            flash("Username must be an email ending with @pupils.nlcsjeju.kr")
            return render_template('register.html')

        conn = sqlite3.connect('static/database.db')
        cursor = conn.cursor()

        command = "SELECT password FROM Users WHERE username = ?;"
        cursor.execute(command, (username,)) # username = scott123
        result = cursor.fetchone() # None
        if result is None:
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
            command = "INSERT INTO Users (username, password, age, gender) VALUES (?,?,?,?);"
            cursor.execute(command, (username, hashed_password, age, gender))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        else:
            flash("Existing Username")
            return render_template('register.html')
    else:
        return render_template('register.html')

@app.route('/data', methods=["GET", "POST"])
def data():
    isLogin = False
    isAdmin = False
    if 'username' in session:
        isLogin = True
        isAdmin = checkAdmin(session["username"])
        data = pandas.read_csv("consulting_data_content_logic.csv")
        data = data[data["the_consultant"]=="Human"]
        gender_distribution = data["gender"].value_counts().to_dict() # {"Male": 100,"Female":300","Other":24}
        content_distribution = data["type_of_consulting"].value_counts().to_dict()
        return render_template('data.html',active_page = "data", isLogin=isLogin,isAdmin=isAdmin, gender_distribution=gender_distribution, content_distribution=content_distribution)
    else:
        return redirect(url_for('login'))
@app.route('/get_response', methods=["POST"])
def get_chatbot_response():
    data = request.json
    username = session['username']
    user_input = data["message"]
    response = get_response(user_input) # CHATGPT
    #topic = get_intent(user_input)  #BERT

    #logging
    current_date = datetime.now().date()
    blob = TextBlob(user_input)
    sentiment = blob.sentiment.polarity # -1 ~ 1
    with sqlite3.connect('static/database.db') as conn:
        cur = conn.cursor() # cursor can execute some SQL command on DB, SQL : Programming for managing DB
        cur.execute('INSERT INTO AI (username, content, date, response,sentiment) VALUES (?, ?, ?, ?,?)',
                    (username, user_input, current_date, response, sentiment ))
        conn.commit()
    # print(response)
    return jsonify({"response": response})

@app.route('/logout', methods=["GET"])
def logout():
    session.clear() # {}
    return redirect(url_for('index'))
@app.route('/my_reservations')
def my_reservations():
    isLogin = True
    if 'username' not in session:
        isLogin = False
        return redirect(url_for('login'))
    return render_template('myreservations.html', isLogin = isLogin)
@app.route('/api/getReservations')
def get_reservations():
    with sqlite3.connect('static/database.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT title, start, end, description, username FROM Reservations')
        reservations = cursor.fetchall()

    reservations_list = [
        {
            'title': row[0],
            'start': row[1],
            'end': row[2],
            'description': row[3],
            'username': row[4]
        }
        for row in reservations
    ]
    return jsonify(reservations_list)

@app.route('/api/saveReservation', methods=['POST'])
def save_reservation():
    title = request.form['title']
    description = request.form['description']
    start = request.form['start']
    end = request.form['end']
    username = session.get('username', 'anonymous')  # Replace with actual user logic

    if username is None:
        return jsonify({'status': 'error', 'message': 'User not found'})

    with sqlite3.connect('static/database.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO Reservations (title, start, end, description, username) VALUES (?, ?, ?, ?, ?)
        ''', (title, start, end, description, username))
        conn.commit()

    return jsonify({'status': 'success'})
if __name__ == '__main__':
    app.run(debug=True, port=5000)