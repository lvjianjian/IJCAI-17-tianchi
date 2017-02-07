#!/usr/bin
# -*- coding: utf-8 -*-

import urllib2
import datetime
def judgeHoliday(query):
    """
        @query a single date: string eg."20160404"
        @return day_type: 0 workday -1 holiday
        """
    url = 'http://www.easybots.cn/api/holiday.php?d=' + query
    req = urllib2.Request(url)
    resp = urllib2.urlopen(req)
    content = resp.read()
    if(content):
        # 20161001:2 20161002:2 20161003:2 20161004:1
        # "0"workday, "1"leave, "2"holiday
        day_type = content[content.rfind(":")+2:content.rfind('"')]
        if day_type == '0':
            return 0
        else:
            return -1

if __name__ == "__main__":
    start = datetime.datetime(2015,6,1)
    end = datetime.datetime(2016,10,31)
    xDay = (end - start).days
    aDay = datetime.timedelta(days=1)
    i = 0
    file_object = open('data/holiday.csv', 'w')
    while i <= xDay:
        time = start.strftime('%Y%m%d')
        start += aDay
        i += 1
        file_object.write("%s,%d\n" % (time,judgeHoliday(time)))
    file_object.close()