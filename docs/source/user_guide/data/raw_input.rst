.. _data-rawdata:

Raw Input
==================

SATGL supports two data input formats: CNF files and AIG files. You can build the graph with the following functions:

CNF Files
-------------------

+---------------------+---------------------------------------------------------------+
| Function            | Description                                                   |
+=====================+===============================================================+
| ``parse_cnf_file``  | Parses a CNF (Conjunctive Normal Form) file and extracts      |
|                     | information such as the number of variables, number of        |
|                     | clauses, and the list of clauses.                             |
+---------------------+---------------------------------------------------------------+
| ``build_hetero_lcg``| Constructs a heterogeneous graph representing a CNF formula   |
|                     | using the Literal-Clause Graph (LCG) model. The LCG is a      |
|                     | bipartite graph representation of CNF formulas.               |
+---------------------+---------------------------------------------------------------+
| ``build_hetero_vcg``| Constructs a heterogeneous graph representing a CNF formula   |
|                     | using the Variable-Clause Graph (VCG) model. The VCG is a     |
|                     | bipartite graph representation focusing on variables and      |
|                     | clauses.                                                      |
+---------------------+---------------------------------------------------------------+
| ``build_homo_lcg``  | Constructs a homogeneous graph representing a CNF formula     |
|                     | using the Literal-Clause Graph (LCG) model. The LCG is a      |
|                     | bipartite graph representation of CNF formulas.               |
+---------------------+---------------------------------------------------------------+
| ``build_homo_vcg``  | Constructs a homogeneous graph representing a CNF formula     |
|                     | using the Variable-Clause Graph (VCG) model. The VCG is a     |
|                     | bipartite graph representation focusing on variables and      |
|                     | clauses.                                                      |
+---------------------+---------------------------------------------------------------+
| ``build_homo_vig``  | Constructs a homogeneous graph representing a CNF formula     |
|                     | using the Variable-Instance Graph (VIG) model. The VIG        |
|                     | captures relationships between variables and their instances. |
+---------------------+---------------------------------------------------------------+
| ``build_homo_lig``  | Constructs a homogeneous graph representing a CNF formula     |
|                     | using the Literal-Instance Graph (LIG) model. The LIG         |
|                     | captures relationships between literals and their instances.  |
+---------------------+---------------------------------------------------------------+

AIG Files
-------------------

+--------------+--------------------------------------------------------------+
| Function     | Description                                                  |
+==============+==============================================================+
| ``build_aig``| Constructs a DGL graph from an AIG file by converting CNF to |
|              | AIG format, optimizing using ABC commands, and parsing AAG   |
|              | format. Adds node attributes like type and level.            |
+--------------+--------------------------------------------------------------+


