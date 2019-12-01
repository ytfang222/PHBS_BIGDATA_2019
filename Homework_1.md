## Homework_1

Name：Fang Yuting 

Major：19 Fintech

Student ID：1901212576 

### 1.An interesting big data problem

------

**1.1 Define the problem**

As the Internet of Things and other data acquisition and generation technologies advance, data being generated is growing at an exponential rate at all scales in many online and scientific platforms. 

Zhihu, as a Knowledge Q& A website, has a strong social attribute. Zhihu shares knowledge, experience, and different perspectives of each person in different life time axes. A large number of “zhihu”users are engaged in the Internet and computer software industry. 

This report want to explore user information and interpersonal topological relationship and find the intricsic value behind the information. The problem here is defined as: how to quickly find common interests and hobbies for unfamiliar participants and efficiently identify  high value users (bigV).

Firstly, we want to know the basic information of each person. Then we want to get the followers and the interesting topic the person concerned. Finally, we want to find out the intrinsic value of these information which is beneficial for future use.

**1.2 The intrinsic big data properties**

- Volume

Tens of thousands of users will produce large amount of data  on social networks everyday. In June 2018, Zhihu has provided 15000 knowledge service products, with 5000 producers and 6 million Zhihu paying users. Every day, more than one million people use Zhihu University. On December 13, 2018, the number of users exceeded 220 million, a year-on-year increase of 102%. Every day, these users contribute to billions of images, posts, videos, tweets etc. We can now imagine the insanely large amount data that is generated every minute and every hour.  

- Variety

Variety in Big Data refers to all the structured and unstructured data that has the possibility of getting generated either by humans. The most commonly added data in social media are structured-texts, pictures & videos. However, unstructured data like emails, voicemails, hand-written text, audio recordings etc, are also important elements under Variety. Variety is all about the ability to classify the incoming data into various categories.

- Velocity

With Velocity we refer to the speed with which data are being generated. Users can get the information at the first time. Social information is widely and quickly spread every day. This is like a nuclear data explosion. Big Data helps the “zhihu.com” to hold this explosion, accept the incoming flow of data and at the same time process it fast so that it does not create bottlenecks. 

### 2.Workflow to solve the problem

------

**2.1 Oversee the workflow**

![images](https://github.com/ytfang222/PHBS_BIGDATA_2019/raw/master/1.png)
![images](https://github.com/ytfang222/PHBS_BIGDATA_2019/raw/master/2.png)

**2.2 Data Source**

Using the Python's scrape framework to get the basic information of 300W + core users. The data is in JSON format. The information that we see in the web page is as follows：
![images](https://github.com/ytfang222/PHBS_BIGDATA_2019/raw/master/3.png)
**2.3 Database**

The file stores data to mangodb, which clearly defines user attributes and user behavior data, including name, gender, location and other information list.

The information that we need to store in the database is as follows:

| Self-information list | Meaning                             |
| --------------------- | ----------------------------------- |
| avator_url            | User image URL                      |
| token                 | User ID                             |
| Name                  | User Name                           |
| headling              | One sentence introduction           |
| gender                | Male or female                      |
| location              | Place of residence                  |
| topic                 | Interesting topics concerned        |
| employments           | Working experience                  |
| educations            | Education experience                |
| followingCount        | The number of following             |
| followerCount         | The number of follower              |
| questionCount         | Number of questions raised by users |
| answerCount           | Number of answers raised by users   |
| VoteupCount           | Number of users approved            |

| **Interest list** | **Meaning**                 |
| ----------------- | --------------------------- |
| most_good_topic   | Topics of greatest interest |
| following         | Users he focuses on         |
| business          | Woking industry             |
| answer            | The question he answers     |

**2.4 Data Analysis**

We use pandas in python to filter the invalid data and analyze the data. We also use matplotlib to visualize the data.

At the qualitative level, the analysis shows the generalization of the users' individual nature and characteristics. We can form each user some specific label.

At the quantitative level, each label is given a specific weight, and the total label weight is calculated to describe a person accurately. This label can be used to make user profile and predict the user' perference. First, the Input is the result of user portrait and content features. Second, use computational logic to transform these content features into likability according to certain rules. Third, sort content from high to low like as output. Finally, solve the problem whick is finding the content that users like from the massive content.

**2.5 Draw Conslusion**

Through we analysis the users' information , we can summary the information as user profile and user behavior. The results of analysis can  bring high value in improving the recommendation system and expand the influence of the 'zhihu.com' community. 

- **Accurately recommend:** "Zhihu.com" have accumulated a certain amount of users and data, we should give priority to analysis the users background. This is a way called user's portrait. It is based on all kinds of data left by users on the Internet, actively or passively collected, and finally processed into a series of labels. We can accurately recommend content to users through user portraits and create a better ecosystem for each user. We can also recommend suitable users and strengthen people's contact. If the user thinks the recommended content is good, it will trigger the reward function.
- **Get high-value content:** We classify users accurately by analyzing their behavior. In some specific areas, users can find a sense of common belonging. When users are in a common interest group, users will be more likely to bring a variety of valuable content to the community. Users' recognition of high-value users(big V) will also bring high-quality content.
- **Get new users：**After we analysis the big data of the users, we can do a series of optimization, feedback, adjustment and future get more users. “Zhihu.com” should let old users feel that the value of the community is high enough, so they are willing to invitate new users. In that way, “zhihu.com” community can expand network influence.

### 3.Databases to be used

------

We uses NoSQL(non relational database) to store and read data from Mangodb.

**3.1 why NoSQL suitable for this big data problem**

NoSQL is suitable for storing unstructured data, such as articles and comments:

1. These data are usually used for fuzzy processing, such as full-text search. It's only suitable for storing simple data.
2. These data are massive, and the growth rate is unpredictable.
3. Getting data by key is very efficient, but the support for join or other structured queries is relatively poor. Each tuple can have different fields, and each tuple can add some key value pairs as needed, so it will not be limited to fixed structure, and it can reduce some time and space overhead.

**3.2 Why MongoDB**

Firstly, MongoDB is talked in the class. 

Second, MongoDB is a highly flexible and scalable NoSQL database management platform that is document-based, can accommodate different data models.It was developed as a solution for working with large volumes of distributed data that cannot be processed effectively in relational models, which typically accommodate rows and tables. 

Third, MongoDB is free and open-source.

**Advantages of MongDB**

1. When using relational databases, we need several tables for a construct. With Mongo’s document-based model, we can represent a construct in a single entity, especially for immutable data.  
2. The query language used by MongoDB supports dynamic querying.
3. The schema in MongoDB is implicit, meaning we do not have to enforce it. This makes it easier to represent inheritance in the database in addition to improving polymorphism data storage.
4. Horizontal storage makes it easy to scale.

**Limitations of MongoDB**

1. To use joins, we have to manually add code, which may cause slower execution and less-than-optimum performance.
2. Lack of joins also means that MongoDB requires a lot of memory as all files have to be mapped from disk to memory.

### 4.Reference

------

1.https://www.mongodb.com/what-is-mongodb
2.https://docs.mongodb.com/manual/
3.https://www.zhihu.com
4.www.datastax.com/NoSQL/Guide‎
5. Application of Workflow Technology for Big Data Analysis Service 
