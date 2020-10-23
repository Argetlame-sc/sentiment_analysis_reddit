import json
import praw
import psaw
import re
import argparse
import datetime as dt
import signal
import os

from reddit_dataset import load_all_comments, save_comments

def retrieve_flair(api, raw_comments_list, num_submission, flair_list, skip_condition, limit=None, after_cond=None):
    try:
        for submission in api.search_submissions(subreddit="starcitizen", after=after_cond):
            if not submission.link_flair_text in flair_list:
                continue
            if skip_condition(submission):
                continue
            if not limit is None and len(raw_comments_list) >= limit:
                break

            print("submission {} - flair {}".format(submission.title, submission.link_flair_text))
            submission.comments.replace_more(limit=None, threshold=1)
            raw_comments_list += submission.comments.list()

            num_submission += 1
    except KeyboardInterrup:
        print("Interrupting retrieve_flair")
    finally:
        return raw_comments_list, num_submission

def collect_comment_star_citizen(save_file, limit=None, used_saved=False, append=False,
                                do_roadmap=True, flair_list=["OFFICIAL"]):

    if used_saved and not append:
        try:
            comments_list = load_all_comments(db_name=save_file)
            return comments_list
        except:
            print("Could not retrieve saved comments, getting comment normally")
    else:
        comments_list = []

    with open("credentials.json") as f:
        credentials = json.loads(f.read())

    reddit = praw.Reddit(client_id=credentials["id"],
                     client_secret=credentials["secret"],
                     user_agent="Comment Extraction")
    api = psaw.PushshiftAPI(reddit)

    title_filter = re.compile(".*(Star Citizen Roadmap Update|Squadron 42 Roadmap Update).*")
    official_title_filter = re.compile(".*([Ee]vocati +[Pp]atch|([Pp]atch|Release|P[Tt][Uu])[ -]+[nN]otes).*")

    initial_epoch = int(dt.datetime(2012, 10, 20).timestamp())

    raw_comments_list = []
    num_submission = 0

    try:
        if do_roadmap:
            for submission in api.search_submissions(author="Odysseus-Ithaca", subreddit="starcitizen", after=initial_epoch):
                if submission is None:
                    break
                if title_filter.match(submission.title) is None:
                    continue

                print("submission {} - flair {}".format(submission.title, submission.link_flair_text))
        
                submission.comments.replace_more(limit=None, threshold=1)
                raw_comments_list += submission.comments.list()

                num_submission += 1
                if not limit is None and len(raw_comments_list) >= limit:
                    break

            print("{} submission for odysseus done".format(num_submission))

        if len(flair_list) > 0:

            raw_comments_list, num_submission = retrieve_flair(api, raw_comments_list, num_submission, flair_list,
                                                        lambda s:official_title_filter.match(s.title), after_cond=initial_epoch)

    except KeyboardInterrup:
        print("Received keyboard interrup - stopping scraping")
    finally:
        print("{} submission done in total".format(num_submission))


        #Retrieve all attributes
        tmp_list = []
        for c in raw_comments_list:
            attributes_raw = vars(c)
            attributes = {}
            #filter lazy attributes
            attributes["submission_title"] = c.submission.title
            attributes["submission_name"] = c.submission.name
            attributes["submission_flair"] = c.submission.link_flair_text
            for key, value in attributes_raw.items():
                if key.startswith("_"):
                    continue
                elif key == "subreddit" or key == "author":
                    continue

                attributes[key] = value

            tmp_list.append(attributes)

        comments_list += tmp_list

        #Cache results
        save_comments(comments_list, append, db_name=save_file)

        return comments_list

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_file', default="comments_saved.db", help="Filename for saving results into a sqlite databse)
    parser.add_argument('--use_saved_data', default=False, action='store_true', help="Use saved comments")
    parser.add_argument('--append_saved_data', default=False, action='store_true', help="Use saved comments and append results to it")
    parser.add_argument('--no_roadmap', action='store_false', help="Do not retrieve data from roadmap update")
    parser.add_argument('--retrieve_flairs', nargs='+', default=["OFFICIAL"], help="List of flair to retrieve")
    parser.add_argument('--len_random_dataset', default=200, type=int, help="Size of random dataset extracted")

    return parser

if __name__ == "__main__":

    parser = common_arg_parser()
    args = parser.parse_args()
    print(args.retrieve_flairs)

    comments_list = collect_comment_star_citizen(args.save_file, used_saved=args.use_saved_data, append=args.append_saved_data,
                                                do_roadmap=args.no_roadmap, flair_list=args.retrieve_flairs)

    num_comments = len(comments_list)

    print("num comments {}".format(num_comments))

    import numpy as np

    indices = np.arange(num_comments)
    np.random.shuffle(indices)
    indices = indices[:args.len_random_dataset]

    import pandas as pd
    review = np.zeros((args.len_random_dataset,))

    txt = []
    for idx in indices:
        txt.append(comments_list[idx]["body"])

    df = pd.DataFrame(data={"review": review, "text": txt, "id": indices})

    df.to_csv("random_starcitizen_dataset.csv", index=False)


