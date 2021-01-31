.. role:: raw-html-m2r(raw)
   :format: html


Hierarchical panoptic format and labels encoding
================================================

:raw-html-m2r:`<img src="../../readme/hierarchical_format.jpg" height="300"/>`

Each pixel in our hierarchical label format has an up to 7-digit *universal id* (\ *uid*\ ) containing:


* An up to 2-digit *semantic id* (\ *sid*\ ), encoding the semantic-level *things* or *stuff* class.
* An up to 3-digit *instance id* (\ *iid*\ ), a counter of instances per *things* class and per image. This is optional.
* An up to 2-digit *part id* (\ *pid*\ ), encoding the parts-level semantic class per-instance and per-image. This is optional, but if provided requires also an *iid*. Only *things* parts are covered by this format.

We compactly encode the aforementioned *ids* (\ *sid*\ , *iid*\ , *pid*\ ) into an up to 7-digit *uid*. Starting from the left, the first one or two digits encode the semantic class, the next 3 encode the instance (after zero pre-padding), and the final two encode the parts class (after zero pre-padding).

Using the above encoding:


* 1-2 digit *uids* encode only semantic-level labels
* 4-5 digit *uids* encode semantic-instance-level labels
* 6-7 digit *uids* encode semantic-instance-parts-level labels

For example, in *Cityscapes-Panoptic-Parts*\ , a *sky* (\ *stuff*\ ) pixel will have *uid* = 23, a *car* (\ *things*\ ) pixel that is labeled only on the semantic level will have *uid* = 26, if it's labeled also on instance level it can have *uid* = 26002, and a *person* (\ *things*\ ) pixel that is labeled on all three levels can have *uid* = 2401002.

..

   The format does not cover parts-level classes for *stuff* semantic classes for now.


Unlabeled pixels
----------------

We handle the unlabeled / void / "do not care pixels" in the three levels as follows:


* Semantic level: For *Cityscapes-Panoptic-Parts* we use the original Cityscapes void class. For *PASCAL-Panoptic-Parts* we use the class with *uid* = 0.
* Instance level: For instances the void class is not required. If a pixel does not belong to an object or cannot be labeled on instance level then it has only an up to 2-digit *semantic id*.
* Parts level: For both datasets we use the convention that, for each semantic class, the part-level class with *pid* = 0 represents the void pixels, e.g., for a *person* pixel, *uid* = 2401000 represents the void parts pixels of instance 10. The need for a void class arises during the manual annotation process but in principle it is not needed at the parts level. Thus, we try to minimize void parts level pixels and assign them instead only the semantic- or semantic-instance -level labels.
