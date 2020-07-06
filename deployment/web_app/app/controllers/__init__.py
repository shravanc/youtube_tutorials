import os
from flask import Blueprint, current_app
from flask_bootstrap import Bootstrap

from app.controllers.admission_controller import index as admission_index

template_dir = os.path.abspath('app/views/admission')

admission_blueprints = Blueprint('admission', 'api', template_folder=template_dir)
admission_blueprints.add_url_rule('/', view_func=admission_index, methods=['GET', 'POST'])
