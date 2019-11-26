#!/usr/bin/env python3
from os import path, system
import pandas as pd

if not path.exists('RWAS/data.csv'):
    system("wget https://openpsychometrics.org/_rawdata/RWAS.zip")
    system("unzip RWAS.zip")


item_text = ["The established authorities generally turn out to be right about things, while the radicals and protestors are usually just \"loud mouths\" showing off their ignorance.",
             "Women should have to promise to obey their husbands when they get married.",
             "Our country desperately needs a mighty leader who will do what has to be done to destroy the radical new ways and sinfulness that are ruining us.",
             "Gays and lesbians are just as healthy and moral as anybody else.",
             "It is always better to trust the judgement of the proper authorities in government and religion than to listen to the noisy rabble-rousers in our society who are trying to create doubt in people's minds.",
             "Atheists and others who have rebelled against the established religions are no doubt every bit as good and virtuous as those who attend church regularly.",
             "The only way our country can get through the crisis ahead is to get back to our traditional values, put some tough leaders in power, and silence the troublemakers spreading bad ideas.",
             "There is absolutely nothing wrong with nudist camps.",
             "Our country needs free thinkers who have the courage to defy traditional ways, even if this upsets many people.",
             "Our country will be destroyed someday if we do not smash the perversions eating away at our moral fiber and traditional beliefs.",
             "Everyone should have their own lifestyle, religious beliefs, and sexual preferences, even if it makes them different from everyone else.",
             "The \"old-fashioned ways\" and the \"old-fashioned values\" still show the best way to live.",
             "You have to admire those who challenged the law and the majority's view by protesting for women's abortion rights, for animal rights, or to abolish school prayer.",
             "What our country really needs is a strong, determined leader who will crush evil, and take us back to our true path.",
             "Some of the best people in our country are those who are challenging our government, criticizing religion, and ignoring the \"normal way things are supposed to be done.\"",
             "God's laws about abortion, pornography and marriage must be strictly followed before it is too late, and those who break them must be strongly punished.",
             "There are many radical, immoral people in our country today, who are trying to ruin it for their own godless purposes, whom the authorities should put out of action.",
             "A \"woman's place\" should be wherever she wants to be. The days when women are submissive to their husbands and social conventions belong strictly in the past.",
             "Our country will be great if we honor the ways of our forefathers, do what the authorities tell us to do, and get rid of the \"rotten apples\" who are ruining everything.",
             "There is no \"one right way\" to live life; everybody has to create their own way.",
             "Homosexuals and feminists should be praised for being brave enough to defy \"traditional family values.\"",
             "This country would work a lot better if certain groups of troublemakers would just shut up and accept their group's traditional place in society."]

reverse = [
    3, 5, 7, 8, 10, 12, 14, 17, 19, 20
]

def get_data(reorient=False):
    if not path.exists('RWAS/data.csv'):
        system("wget https://openpsychometrics.org/_rawdata/RWAS.zip")
        system("unzip RWAS.zip")
    data = pd.read_csv('RWAS/data.csv', low_memory=False)
    item_responses = data.loc[:, map(lambda x: 'Q'+str(x), list(range(1, 23)))]
    if reorient:
        item_responses.iloc[:, reverse] = 9 - item_responses.iloc[:, reverse]
    # system("rm -r RWAS")
    return item_responses
