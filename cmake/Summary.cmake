################################################################################################
# Caffe status report function.
# Automatically align right column and selects text based on condition.
# Usage:
#   caffe_status(<text>)
#   caffe_status(<heading> <value1> [<value2> ...])
#   caffe_status(<heading> <condition> THEN <text for TRUE> ELSE <text for FALSE> )
function(caffe_status text)
  set(status_cond)
  set(status_then)
  set(status_else)

  set(status_current_name "cond")
  foreach(arg ${ARGN})
    if(arg STREQUAL "THEN")
      set(status_current_name "then")
    elseif(arg STREQUAL "ELSE")
      set(status_current_name "else")
    else()
      list(APPEND status_${status_current_name} ${arg})
    