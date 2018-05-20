from flask import Flask, render_template, request
from wtforms import Form, FloatField, validators
from web_app.flask_django_tutorial.web4sciapps.doc.src.web4sa.src_web4sa.apps.flask_apps.hw1.compute import compute

app = Flask(__name__)


# Model
class InputForm(Form):
    r = FloatField(validators=[validators.InputRequired()])


# View
@app.route('/hw2', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        r = form.r.data
        s = compute(r)
    else:
        s = None

    return render_template("view.html", form=form, s=s)


if __name__ == '__main__':
    app.run(debug=True)
