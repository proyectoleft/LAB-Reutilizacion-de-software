from flask import Flask, render_template, redirect, url_for, request
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    mensaje = request.args.get('mensaje')
    return render_template('index.html', mensaje=mensaje)

@app.route('/corners')
def definir_corners():
    subprocess.run(['python', 'board_segmentation.py'])
    return redirect(url_for('index', mensaje='Esquinas definidas correctamente.'))

@app.route('/deteccion')
def detectar():
    subprocess.run(['python', 'deteccion.py'])
    return redirect(url_for('index', mensaje='Detección de piezas completada.'))

@app.route('/tablero')
def tablero():
    subprocess.run(['python', 'main.py'])
    return redirect(url_for('index', mensaje='Tablero mostrado correctamente.'))

if __name__ == '__main__':
    app.run(debug=True)
