/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                            Orange4Home DATASET
                            
   
   Julien Cumin [1,2], Grégoire Lefebvre [1], Fano Ramparany [1], James L. Crowley [2]
   
   [1] Orange Labs, France.
       {julien1.cumin, gregoire.lefebvre, fano.ramparany}@orange.com
       
   [2] Univ. Grenoble Alpes, Inria, CNRS, Grenoble INP, LIG, F-38000 Grenoble, France.
       james.crowley@inria.fr
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */


The Orange4Home Dataset is a dataset of routines of daily living captured in an instrumented smart home.

This dataset is the result of a joint work between Orange and Inria.

This work benefited from the support of the French State through the Agence Nationale de la Recherche under the Future Investments program referenced ANR-11-EQPX-0002.



/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   DATA
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
Data can be found in a single csv file: "o4h_all_events".


Data can be also found in a MySQL export format in the "mysql_dump" directory (238 files).
This directory is a dump of a MySQL database, which can be reimported in a MySQL system.

When reimporting this database dump, the schema name must be "openhab".

Each table corresponds to a sensor, except for "openhab.items".
"openhab.items" contains the correspondence between table names and sensor names.

Table "openhab.item400" contains the labels of activities.

Each sensor's name begins with the name of the place it is located in.



/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   OTHER FILES
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

"sensors_localisation.txt" contains the list of sensors per place, with their domain of values.

"ground_floor.pdf" and "first_floor.pdf" show the simplified layout of the apartment, annotated with the name of places.

"planning.pdf" contains the 4 weeks planning of activties given to the occupant.



/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   CONDITIONS OF USE
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

We recommend to call this dataset the "Orange4Home dataset".

You must acknowledge the use of this dataset in your publications by referencing the following publication:

    Julien Cumin, Grégoire Lefebvre, Fano Ramparany, and James L. Crowley. 
    "A Dataset of Routine Daily Activities in an Instrumented Home". 
    In 11th International Conference on Ubiquitous Computing and Ambient Intelligence (UCAmI), 2017.

