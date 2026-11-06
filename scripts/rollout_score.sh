#!/bin/bash
# usage: rollout_score.sh ref.txt arm.txt -> mean matched prefix, token agreement, all-match prompts (one line of token ids per prompt)
paste -d'|' "$1" "$2" | awk -F'|' '{n=split($1,a," ");split($2,b," ");p=0;stop=0;m=0;for(i=1;i<=n;i++){if(a[i]==b[i]){m++;if(!stop)p++}else stop=1};sp+=p;sm+=m;st+=n;if(m==n)al++;np++} END{printf "prefix %.2f agreement %.1f%% allmatch %d/%d\n",sp/np,100*sm/st,al,np}'
