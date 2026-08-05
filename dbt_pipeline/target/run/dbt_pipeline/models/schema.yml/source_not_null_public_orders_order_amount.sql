
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select order_amount
from "postgres"."public"."orders"
where order_amount is null



  
  
      
    ) dbt_internal_test